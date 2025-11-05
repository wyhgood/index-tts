import io
import os
import time
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# —— 你的 TTS 模型导入 —— #
from indextts.infer_v2 import IndexTTS2

# 环境准备：确保工作目录含有 checkpoints、examples 等
# 你也可以通过环境变量覆盖默认路径
CFG_PATH = os.environ.get("INDEXTTS_CFG", "checkpoints/config.yaml")
MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "checkpoints")
DEFAULT_SPK = os.environ.get("INDEXTTS_DEFAULT_SPK", "examples/voice_01.wav")

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
    # 启动即失败，直接抛出，便于容器/进程管理器重启
    raise RuntimeError(f"Failed to load TTS model: {e}")

# 简单的互斥锁，避免并发下显卡/模型状态问题（如需高并发可改队列/多进程）
_infer_lock = threading.Lock()

app = FastAPI(title="IndexTTS2 HTTP Service", version="1.0.0")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    spk_audio_prompt_path: Optional[str] = Field(
        default=None,
        description="可选，说话人音色参考音频路径。默认使用 examples/voice_01.wav"
    )
    verbose: Optional[bool] = Field(default=False, description="是否打印详细日志")


@app.get("/", response_class=PlainTextResponse)
def root():
    return "IndexTTS2 is running. POST /synthesize with JSON {'text': '...'}"


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` is required and cannot be empty.")

    spk_path = req.spk_audio_prompt_path or DEFAULT_SPK
    if not os.path.isfile(spk_path):
        raise HTTPException(status_code=400, detail=f"Speaker prompt file not found: {spk_path}")

    # 生成到临时文件，然后以流方式返回
    try:
        with _infer_lock:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_wav = os.path.join(tmpdir, "gen.wav")
                t0 = time.time()
                tts.infer(
                    spk_audio_prompt=spk_path,
                    text=text,
                    output_path=out_wav,
                    verbose=bool(req.verbose),
                )
                dur_ms = int((time.time() - t0) * 1000)

                if not os.path.isfile(out_wav) or os.path.getsize(out_wav) == 0:
                    raise RuntimeError("TTS generated empty output.")

                # 读入内存后再返回（临时目录即将删除）
                with open(out_wav, "rb") as f:
                    audio_bytes = f.read()

        # 流式返回 WAV；同时给一个下载文件名
        filename = f"tts_{int(time.time())}.wav"
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Generation-Time-ms": str(dur_ms),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        # 捕获并返回 JSON 错误
        return JSONResponse(
            status_code=500,
            content={"error": "synthesis_failed", "message": str(e)},
        )
