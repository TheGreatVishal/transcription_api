import os
import uuid

BASE_CACHE = "/tmp/hf_cache"

os.environ["HF_HOME"] = BASE_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = BASE_CACHE
os.environ["TRANSFORMERS_CACHE"] = BASE_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import shutil
import time

app = FastAPI(root_path="")

model = None

# 🚀 LOAD MODEL ON STARTUP
@app.on_event("startup")
def load_model():
    global model
    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )


@app.get("/")
def home():
    return {"status": "Whisper API running"}

@app.get("/ping")
def ping():
    return {"msg": "pong"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        # 💾 SAVE FILE
        file_path = f"temp_{uuid.uuid4()}.wav"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)


        # 🔥 STEP 1: AUTO DETECTION
        segments, info = model.transcribe(
            file_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # 🔥 STEP 2: FALLBACK TO HINDI IF LOW CONFIDENCE
        if info.language_probability < 0.6:
            segments, info = model.transcribe(
                file_path,
                language="hi"
            )

        # 🧠 BUILD TEXT
        texts = []
        for i, segment in enumerate(segments):
            texts.append(segment.text)

        final_text = " ".join(texts).strip()

        # 🛑 HANDLE EMPTY AUDIO
        if not final_text:
            final_text = "No speech detected"

        # ⏱️ TIME
        total_time = time.time() - start_time
    
        return {
            "success": True,
            "text": final_text,
            "language": info.language
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)