"""
V-Translator 后端服务 (Render 优化版)
- 使用 faster-whisper 替代原生 whisper 以节省内存
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import httpx
import tempfile
import os
import uvicorn

app = FastAPI(title="V-Translator API")

# 允许所有来源（手机App访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 加载 Faster-Whisper 模型 ──────────────────────────
# 使用 int8 量化进一步压缩内存占用
print("正在加载 Faster-Whisper 模型 (tiny)...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("Whisper 模型加载完成 ✅")

# 语言代码映射
LANG_MAP = {
    "vn": {"whisper": "vi", "libre_src": "vi", "libre_tgt": "zh"},
    "cn": {"whisper": "zh", "libre_src": "zh", "libre_tgt": "vi"},
}

@app.get("/")
def health():
    return {"status": "ok", "message": "V-Translator API 运行中 (Faster-Whisper版)"}

@app.post("/recognize")
async def recognize(
    audio: UploadFile = File(...),
    src_lang: str = Form(default="vn"),
):
    if src_lang not in LANG_MAP:
        raise HTTPException(status_code=400, detail="src_lang 必须是 'vn' 或 'cn'")

    lang_cfg = LANG_MAP[src_lang]
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 2️⃣ Faster-Whisper 语音识别
        segments, info = whisper_model.transcribe(
            tmp_path, 
            language=lang_cfg["whisper"],
            beam_size=5
        )
        
        # 将识别到的片段合并为完整文本
        recognized_text = "".join([segment.text for segment in segments]).strip()

        if not recognized_text:
            return {"recognized": "", "translated": "", "error": "未能识别语音"}

        # 3️⃣ 翻译
        translated_text = await translate(
            text=recognized_text,
            src=lang_cfg["libre_src"],
            tgt=lang_cfg["libre_tgt"],
        )

        return {
            "recognized": recognized_text,
            "translated": translated_text,
            "error": None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

async def translate(text: str, src: str, tgt: str) -> str:
    servers = [
        "https://libretranslate.de/translate",
        "https://translate.argosopentech.com/translate",
        "https://libretranslate.com/translate",
    ]
    payload = {
        "q": text,
        "source": src,
        "target": tgt,
        "format": "text",
        "api_key": "",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in servers:
            try:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    return res.json().get("translatedText", text)
            except:
                continue
    return f"[翻译服务忙] {text}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server_main:app", host="0.0.0.0", port=port)
