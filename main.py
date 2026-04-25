import os, uuid, json, asyncio, base64
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from PIL import Image, ImageOps
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NOTION_TOKEN      = os.environ["NOTION_TOKEN"]
NOTION_DB_ID      = os.environ["NOTION_DB_ID"]
NOTION_HEADERS    = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}

jobs: dict[str, dict] = {}

# ------------------------------------------------------------------ helpers

def fix_and_resize(image_bytes: bytes, max_size: int = 1500) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def analyze_image(image_bytes: bytes) -> tuple[str, str, str]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    b64 = base64.standard_b64encode(image_bytes).decode()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": (
                    "この書類画像を日本語で解析してください。\n"
                    "以下のJSON形式のみで返答してください。余分なテキスト不要。\n"
                    '{"title":"書類タイトル（具体的に）","summary":"内容の要約（100字以内）",'
                    '"category":"カテゴリ（役所・手続き／医療・保険／学費・奨学金／給与・規程／金融・税務／保険・年金／その他のいずれか）",'
                    '"text":"書類の全文テキスト（読み取れる文字を全て書き起こす）"}'
                )},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return data["title"], data["summary"], data["category"], data.get("text", "")


def upload_to_notion(image_bytes: bytes, filename: str) -> str:
    with httpx.Client(timeout=60) as client:
        r1 = client.post(
            "https://api.notion.com/v1/file_uploads",
            headers=NOTION_HEADERS,
            json={"filename": filename, "content_type": "image/jpeg"},
        )
        r1.raise_for_status()
        file_id    = r1.json()["id"]
        upload_url = r1.json()["upload_url"]
        r2 = client.post(
            upload_url,
            headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28"},
            files={"file": (filename, image_bytes, "image/jpeg")},
        )
        r2.raise_for_status()
    return file_id


def create_notion_page(title: str, summary: str, category: str, file_id: str, text: str = "") -> str:
    children = [
        {"object": "block", "type": "image",
         "image": {"type": "file_upload", "file_upload": {"id": file_id}}},
    ]
    if text:
        for chunk_start in range(0, min(len(text), 4000), 2000):
            children.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text[chunk_start:chunk_start+2000]}}]}
            })
    body = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "タイトル":   {"title":     [{"text": {"content": title}}]},
            "AI要約":    {"rich_text": [{"text": {"content": summary}}]},
            "カテゴリ":  {"select":    {"name": category}},
            "ステータス": {"status":   {"name": "未処理"}},
        },
        "children": children,
    }
    with httpx.Client(timeout=60) as client:
        r = client.post("https://api.notion.com/v1/pages", headers=NOTION_HEADERS, json=body)
        r.raise_for_status()
    return r.json()["id"]


# ------------------------------------------------------------------ background job

def process_job(job_id: str, files_data: list[tuple[str, bytes]]):
    job = jobs[job_id]
    job["status"] = "running"
    for i, (filename, raw_bytes) in enumerate(files_data):
        job["current"] = i + 1
        job["current_file"] = filename
        try:
            print(f"[{filename}] 画像リサイズ中...")
            image_bytes = fix_and_resize(raw_bytes)
            print(f"[{filename}] Claude解析中...")
            title, summary, category, text = analyze_image(image_bytes)
            print(f"[{filename}] タイトル={title} カテゴリ={category} テキスト={len(text)}文字")
            print(f"[{filename}] Notionアップロード中...")
            file_id = upload_to_notion(image_bytes, filename)
            page_id = create_notion_page(title, summary, category, file_id, text)
            print(f"[{filename}] 完了 page_id={page_id[:8]}")
            job["results"].append({
                "filename": filename,
                "title":    title,
                "summary":  summary,
                "category": category,
                "page_id":  page_id,
                "success":  True,
            })
        except Exception as e:
            print(f"[{filename}] エラー: {e}")
            job["results"].append({"filename": filename, "error": str(e), "success": False})
    job["status"] = "done"


# ------------------------------------------------------------------ routes

HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="書類スキャン">
<title>書類スキャン</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0a0a0a;--surface:#141414;--border:#222;--accent:#e8e8e8;--muted:#555;--green:#22c55e;--red:#ef4444;--amber:#f59e0b}
  body{background:var(--bg);color:var(--accent);font-family:-apple-system,sans-serif;min-height:100dvh;display:flex;flex-direction:column;padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
  header{padding:24px 20px 8px;border-bottom:1px solid var(--border)}
  h1{font-size:17px;font-weight:600;letter-spacing:.02em}
  .sub{font-size:12px;color:var(--muted);margin-top:4px}
  main{flex:1;padding:20px;display:flex;flex-direction:column;gap:16px}
  .drop-zone{border:1.5px dashed var(--border);border-radius:16px;padding:40px 20px;text-align:center;transition:.2s;position:relative;cursor:pointer}
  .drop-zone:active{border-color:var(--accent);background:#ffffff0a}
  .drop-icon{font-size:40px;margin-bottom:12px}
  .drop-label{font-size:15px;color:var(--muted)}
  .drop-label strong{color:var(--accent)}
  input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
  .preview-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
  .thumb{aspect-ratio:1;border-radius:10px;overflow:hidden;background:var(--surface);position:relative}
  .thumb img{width:100%;height:100%;object-fit:cover}
  .thumb-count{position:absolute;top:4px;right:4px;background:#000a;color:#fff;font-size:10px;padding:2px 5px;border-radius:20px}
  .btn{width:100%;padding:16px;border-radius:14px;border:none;font-size:16px;font-weight:600;cursor:pointer;transition:.15s}
  .btn-primary{background:var(--accent);color:#000}
  .btn-primary:active{opacity:.8}
  .btn-primary:disabled{background:var(--border);color:var(--muted);cursor:default}
  .btn-ghost{background:transparent;color:var(--muted);font-size:14px;padding:12px;margin-top:-8px}
  .progress-card{background:var(--surface);border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:10px}
  .progress-label{font-size:13px;color:var(--muted)}
  .progress-bar-wrap{background:var(--border);border-radius:100px;height:4px}
  .progress-bar{background:var(--accent);border-radius:100px;height:4px;transition:width .4s}
  .result-list{display:flex;flex-direction:column;gap:8px}
  .result-item{background:var(--surface);border-radius:12px;padding:14px 16px;display:flex;gap:12px;align-items:flex-start}
  .result-icon{font-size:18px;flex-shrink:0;margin-top:1px}
  .result-title{font-size:14px;font-weight:500;margin-bottom:3px}
  .result-cat{font-size:12px;color:var(--muted)}
  .result-err{font-size:12px;color:var(--red)}
  .done-header{text-align:center;padding:8px 0}
  .done-header .icon{font-size:36px;margin-bottom:8px}
  .done-header p{font-size:14px;color:var(--muted)}
  .hidden{display:none!important}
</style>
</head>
<body>
<header>
  <h1>書類スキャン</h1>
  <div class="sub">写真を選んで Notion へ自動登録</div>
</header>
<main id="view-select">
  <button class="btn btn-primary" id="addBtn" style="position:relative;overflow:hidden">
    ＋ 写真を追加
    <input type="file" id="fileInput" accept="image/*" style="position:absolute;inset:0;opacity:0;cursor:pointer">
  </button>
  <div class="preview-grid hidden" id="previewGrid"></div>
  <button class="btn btn-primary hidden" id="sendBtn" disabled>Notionへ送信</button>
  <button class="btn btn-ghost hidden" id="clearBtn">クリア</button>
</main>

<main id="view-progress" class="hidden">
  <div class="progress-card">
    <div class="progress-label" id="progressLabel">準備中...</div>
    <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  </div>
  <div class="result-list" id="resultList"></div>
</main>

<main id="view-done" class="hidden">
  <div class="done-header">
    <div class="icon">✅</div>
    <p id="doneMsg"></p>
  </div>
  <div class="result-list" id="resultListDone"></div>
  <button class="btn btn-primary" id="restartBtn" style="margin-top:8px">もう一度送信</button>
</main>

<script>
const $ = id => document.getElementById(id);
let selectedFiles = [];
let pollTimer = null;

$('fileInput').addEventListener('change', e => {
  const newFiles = Array.from(e.target.files);
  selectedFiles = selectedFiles.concat(newFiles);
  e.target.value = '';
  renderPreviews();
});

function renderPreviews() {
  const grid = $('previewGrid');
  grid.innerHTML = '';
  selectedFiles.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const div = document.createElement('div');
    div.className = 'thumb';
    div.innerHTML = `<img src="${url}"><div class="thumb-count">${i+1}</div>`;
    grid.appendChild(div);
  });
  const hasFiles = selectedFiles.length > 0;
  grid.classList.toggle('hidden', !hasFiles);
  $('sendBtn').classList.toggle('hidden', !hasFiles);
  $('clearBtn').classList.toggle('hidden', !hasFiles);
  $('sendBtn').disabled = !hasFiles;
  $('sendBtn').textContent = hasFiles ? `${selectedFiles.length}枚を送信` : 'Notionへ送信';
}

$('clearBtn').addEventListener('click', () => {
  selectedFiles = [];
  $('fileInput').value = '';
  renderPreviews();
});

$('sendBtn').addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files', f, f.name));

  switchView('progress');
  $('progressLabel').textContent = `アップロード中...`;

  const res = await fetch('/upload', {method:'POST', body:fd}).catch(err=>{
    alert('送信エラー: ' + err.message); switchView('select'); return null;
  });
  if (!res) return;
  const {job_id, total} = await res.json();
  poll(job_id, total);
});

function poll(jobId, total) {
  let lastCount = 0;
  pollTimer = setInterval(async () => {
    const r = await fetch(`/status/${jobId}`).catch(()=>null);
    if (!r) return;
    const data = await r.json();

    const done = data.results?.length || 0;
    const pct  = total > 0 ? Math.round(done / total * 100) : 0;
    $('progressBar').style.width = pct + '%';

    if (data.status === 'running' && data.current_file) {
      $('progressLabel').textContent = `処理中 ${data.current}/${total}：${data.current_file}`;
    }

    // 新しい結果を追加
    const list = $('resultList');
    while (lastCount < done) {
      const item = data.results[lastCount];
      list.appendChild(makeResultItem(item));
      lastCount++;
    }

    if (data.status === 'done') {
      clearInterval(pollTimer);
      showDone(data.results, total);
    }
  }, 2000);
}

function makeResultItem(item) {
  const div = document.createElement('div');
  div.className = 'result-item';
  if (item.success) {
    div.innerHTML = `<div class="result-icon">✅</div><div><div class="result-title">${item.title}</div><div class="result-cat">${item.category}</div></div>`;
  } else {
    div.innerHTML = `<div class="result-icon">❌</div><div><div class="result-title">${item.filename}</div><div class="result-err">${item.error||'エラー'}</div></div>`;
  }
  return div;
}

function showDone(results, total) {
  const ok = results.filter(r=>r.success).length;
  $('doneMsg').textContent = `${ok}件 / ${total}件 をNotionに登録しました`;
  const list = $('resultListDone');
  list.innerHTML = '';
  results.forEach(r => list.appendChild(makeResultItem(r)));
  switchView('done');
}

$('restartBtn').addEventListener('click', () => {
  selectedFiles = [];
  $('fileInput').value = '';
  $('previewGrid').innerHTML = '';
  $('resultList').innerHTML = '';
  $('resultListDone').innerHTML = '';
  $('progressBar').style.width = '0%';
  renderPreviews();
  switchView('select');
});

function switchView(name) {
  ['select','progress','done'].forEach(n => {
    $('view-'+n).classList.toggle('hidden', n!==name);
  });
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    files_data = [(f.filename or f"image_{i}.jpeg", await f.read()) for i, f in enumerate(files)]
    jobs[job_id] = {"status": "pending", "results": [], "total": len(files_data), "current": 0, "current_file": ""}
    background_tasks.add_task(process_job, job_id, files_data)
    return {"job_id": job_id, "total": len(files_data)}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return job
