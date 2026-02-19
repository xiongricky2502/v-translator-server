"""
V-Translator 后端服务
- Whisper (openai-whisper): 语音识别（越南语/中文）
- LibreTranslate: 文字翻译
- FastAPI: HTTP 服务框架

部署在 Render.com 免费套餐即可运行
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
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

# ── 加载 Whisper 模型 ──────────────────────────────
# "small" 模型约 244MB，越南语识别效果好，速度可接受
# 可选: "tiny"(75MB,快), "small"(244MB,推荐), "medium"(769MB,最准)
print("正在加载 Whisper 模型，请稍候...")
whisper_model = whisper.load_model("small")
print("Whisper 模型加载完成 ✅")

# LibreTranslate 公共实例（免费，无需Key）
LIBRETRANSLATE_URL = "https://libretranslate.com/translate"

# 语言代码映射
LANG_MAP = {
    "vn": {"whisper": "vi", "libre_src": "vi", "libre_tgt": "zh"},
    "cn": {"whisper": "zh", "libre_src": "zh", "libre_tgt": "vi"},
}


# ── 健康检查 ───────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "V-Translator API 运行中"}


# ── 主接口：语音识别 + 翻译 ─────────────────────────
@app.post("/recognize")
async def recognize(
    audio: UploadFile = File(...),          # 音频文件
    src_lang: str = Form(default="vn"),     # 源语言: "vn" 或 "cn"
):
    if src_lang not in LANG_MAP:
        raise HTTPException(status_code=400, detail="src_lang 必须是 'vn' 或 'cn'")

    lang_cfg = LANG_MAP[src_lang]

    # 1️⃣ 保存音频到临时文件
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 2️⃣ Whisper 语音识别
        result = whisper_model.transcribe(
            tmp_path,
            language=lang_cfg["whisper"],
            fp16=False,  # CPU 模式不用 fp16
        )
        recognized_text = result["text"].strip()

        if not recognized_text:
            return {"recognized": "", "translated": "", "error": "未能识别语音"}

        # 3️⃣ LibreTranslate 翻译
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
        os.unlink(tmp_path)  # 清理临时文件


async def translate(text: str, src: str, tgt: str) -> str:
    """调用 LibreTranslate 公共 API"""
    # 先试免费公共实例
    servers = [
        "https://libretranslate.com/translate",
        "https://translate.argosopentech.com/translate",
        "https://libretranslate.de/translate",
    ]
    payload = {
        "q": text,
        "source": src,
        "target": tgt,
        "format": "text",
        "api_key": "",  # 公共实例不需要 Key
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in servers:
            try:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("translatedText", text)
            except Exception:
                continue
    return f"[翻译服务暂时不可用] {text}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
