import os
import uuid
import time
import shutil

BASE_CACHE = "/tmp/hf_cache"

os.environ["HF_HOME"] = BASE_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = BASE_CACHE
os.environ["TRANSFORMERS_CACHE"] = BASE_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
from huggingface_hub import login

# ✅ HF login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(HF_TOKEN)

app = FastAPI()
model = None

# 🚀 LOAD MODEL
@app.on_event("startup")
def load_model():
    global model
    print("🚀 Loading Whisper model...")
    model = WhisperModel(
        "tiny",   # ⚡ fast + stable
        device="cpu",
        compute_type="int8"
    )
    print("✅ Whisper model ready")


@app.get("/")
def home():
    return {"status": "Whisper API running"}


@app.get("/ping")
def ping():
    return {"msg": "pong"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()
    file_path = None

    try:
        print("\n================= NEW REQUEST =================")

        # 📁 FILE INFO
        print(f"📁 Filename: {file.filename}")
        print(f"📄 Content Type: {file.content_type}")

        # 💾 SAVE FILE
        file_path = f"temp_{uuid.uuid4()}.wav"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"⏱️ File saved at: {file_path}")
        print("🚀 Starting transcription...")

        # 🔥 SINGLE PASS TRANSCRIPTION (FAST)
        segments, info = model.transcribe(
            file_path,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        detected_lang = info.language
        print(f"🎧 Detected Language: {detected_lang}")

        if detected_lang == "hi":
            print("🇮🇳 Hindi detected")
        else:
            print("🌍 Non-Hindi detected")

        # 🧠 BUILD TEXT
        print("\n--- SEGMENTS ---")
        texts = []

        for i, segment in enumerate(segments):
            print(f"[{i}] {segment.start:.2f}s → {segment.end:.2f}s : {segment.text}")
            texts.append(segment.text)

        final_text = " ".join(texts).strip()

        if not final_text:
            final_text = "No speech detected"

        print("\n🧠 FINAL TRANSCRIPTION:")
        print(final_text)

        total_time = time.time() - start_time
        print(f"⏱️ Total Processing Time: {total_time:.2f} sec")
        print("=============================================\n")

        return {
            "success": True,
            "text": final_text,
            "language": detected_lang
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        print(str(e))
        print("=============================================\n")

        return {
            "success": False,
            "error": str(e)
        }

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print("🗑️ Temp file deleted")