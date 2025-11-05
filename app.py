import io
import os
import time
import tempfile
import threading
from typing import Optional
from urllib.parse import urlparse
import hashlib

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# —— 你的 TTS 模型 —— #
from indextts.infer_v2 import IndexTTS2

# 基本路径
CFG_PATH = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
EXAMPLES_DIR = os.environ.get("INDEXTTS_EXAMPLES_DIR", "examples")

# 预加载模型（仅一次）
try:
    t0 = time.time()
    tts = IndexTTS2(
        cfg_path=CFG_PATH,
        model_dir=MODEL_DIR,
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False
    )
    load_ms = int((time.time() - t0) * 1000)
    print(f"[TTS] Model loaded in {load_ms} ms")
except Exception as e:
    raise RuntimeError(f"Failed to load TTS model: {e}")

# 互斥锁，避免并发引起的显卡/状态问题
_infer_lock = threading.Lock()

# 确保 examples 目录存在
os.makedirs(EXAMPLES_DIR, exist_ok=True)

# 允许的音频后缀
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
CONTENT_TYPE_TO_EXT = {
    "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a", "audio/aac": ".m4a",
    "audio/ogg": ".ogg", "application/ogg": ".ogg",
}

# 下载限制
DOWNLOAD_TIMEOUT = (5, 60)  # (连接超时, 读取超时)
MAX_BYTES = 50 * 1024 * 1024  # 50MB

app = FastAPI(title="IndexTTS2 HTTP Service", version="1.1.0")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    spk_audio_url: str = Field(..., description="说话人音色参考音频的 HTTP 地址")
    verbose: Optional[bool] = Field(default=False, description="是否打印详细日志")


@app.get("/", response_class=PlainTextResponse)
def root():
    return "IndexTTS2 is running. POST /synthesize with JSON {'text': '...', 'spk_audio_url': 'http://...'}"


def _safe_filename(name: str) -> str:
    # 只保留字母数字、点、下划线、短横
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in name)


def _guess_ext_from_headers(headers) -> Optional[str]:
    ctype = headers.get("Content-Type", "").split(";")[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(ctype)


def _ensure_local_prompt(spk_audio_url: str) -> str:
    """
    根据传入的 HTTP 地址，检查 examples/ 下是否已有同名文件；
    没有则下载并缓存，返回本地文件路径。
    """
    # 解析 URL，取路径里的 basename
    parsed = urlparse(spk_audio_url)
    base = os.path.basename(parsed.path) or ""

    # 若 URL 没带文件名，就用 URL 的哈希作为名
    if not base or "." not in base:
        h = hashlib.sha1(spk_audio_url.encode("utf-8")).hexdigest()[:16]
        base = f"speaker_{h}.wav"  # 默认 .wav，若后面探测到其他类型会再调整

    base = _safe_filename(base)
    name, ext = os.path.splitext(base)
    ext = ext.lower()

    # 还没确定/非白名单扩展，先占位，稍后根据 Content-Type 再矫正
    if ext not in ALLOWED_EXTS:
        ext = ""

    # 目标路径（先不含扩展）
    target_noext = os.path.join(EXAMPLES_DIR, name)

    # 如果本地已有同名（若没有扩展，尽力匹配任一已存在后缀）
    if ext:
        candidate = target_noext + ext
        if os.path.isfile(candidate):
            return candidate
    else:
        # 尝试匹配任一允许后缀
        for e in ALLOWED_EXTS:
            candidate = target_noext + e
            if os.path.isfile(candidate):
                return candidate

    # 本地没找到，需要下载
    try:
        with requests.get(spk_audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            if r.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download speaker audio: HTTP {r.status_code}")

            # 根据 Content-Type 决定扩展名
            if not ext:
                guessed = _guess_ext_from_headers(r.headers)
                if guessed:
                    ext = guessed
                else:
                    # 无法判断时，兜底 wav
                    ext = ".wav"

            if ext not in ALLOWED_EXTS:
                raise HTTPException(status_code=400, detail=f"Unsupported audio type/extension: {ext}")

            # Content-Length 初步校验
            clen = r.headers.get("Content-Length")
            if clen and int(clen) > MAX_BYTES:
                raise HTTPException(status_code=400, detail=f"File too large (> {MAX_BYTES // (1024*1024)}MB)")

            # 先下载到临时文件，完毕后原子移动
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="dl_", suffix=ext, dir=EXAMPLES_DIR)
            written = 0
            try:
                with os.fdopen(tmp_fd, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > MAX_BYTES:
                            raise HTTPException(status_code=400, detail=f"File too large (> {MAX_BYTES // (1024*1024)}MB)")
                        f.write(chunk)
                final_path = target_noext + ext
                # 若最终名恰好已被其他并发下载占用，直接复用那个
                if os.path.exists(final_path):
                    os.remove(tmp_path)
                    return final_path
                os.replace(tmp_path, final_path)
                return final_path
            except Exception:
                # 发生异常时清理临时文件
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error when downloading: {e}")


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` is required and cannot be empty.")
    if not req.spk_audio_url:
        raise HTTPException(status_code=400, detail="`spk_audio_url` is required.")

    # 获取或下载参考音频到 examples/
    spk_local_path = _ensure_local_prompt(req.spk_audio_url)

    # 生成到临时文件，然后以流方式返回
    try:
        with _infer_lock:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_wav = os.path.join(tmpdir, "gen.wav")
                t0 = time.time()
                tts.infer(
                    spk_audio_prompt=spk_local_path,  # <—— 使用下载/缓存后的本地路径
                    text=text,
                    output_path=out_wav,
                    verbose=bool(req.verbose),
                )
                dur_ms = int((time.time() - t0) * 1000)

                if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
                    raise RuntimeError("TTS generated empty output.")

                with open(out_wav, "rb") as f:
                    audio_bytes = f.read()

        filename = f"tts_{int(time.time())}.wav"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Generation-Time-ms": str(dur_ms),
                "X-Speaker-File": os.path.basename(spk_local_path),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "synthesis_failed", "message": str(e)},
        )
