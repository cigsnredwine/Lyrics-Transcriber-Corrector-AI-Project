# Jayden Nguyen
# @cigsnredwine
# longnguyenhien1@gmail.com
# ========= LYRICS TRANSCRIBER & CORRECTOR AI PROJECT ============

import os
import shutil
import whisper
import torch
import subprocess
import tempfile

# ========== SETTINGS ==========
MODEL_SIZE = "small"   # "tiny", "base", "small", "medium", "large"
CHUNK_LENGTH = 60      # seconds per chunk
AUDIO_DIR = r"G:\Music\FINAL\RỒNG"

RAW_DIR = r"C:\Users\ADMIN\PycharmProjects\TRANSCRIBER\Transcriptions\raw"
CLEAN_DIR = r"C:\Users\ADMIN\PycharmProjects\TRANSCRIBER\Transcriptions\clean"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
# ===============================

# Load Whisper on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model '{MODEL_SIZE}' on {device}...")
model = whisper.load_model(MODEL_SIZE).to(device)

# Build output filename
basename = os.path.splitext(os.path.basename(AUDIO_DIR))[0]
raw_file = os.path.join(RAW_DIR, f"{basename}_raw.txt")
clean_file = os.path.join(CLEAN_DIR, f"{basename}_clean.txt")

# Known "hallucination phrases" to strip
HALLUCINATIONS = [
    "Hãy subscribe cho kênh",
    "Để không bỏ lỡ những video hấp dẫn",
    "hãy đăng ký kênh",
    "subscribe cho kênh",
    "Ghiền Mì Gõ",
    "Thanks for watching guys",
    "La La School"
    "I had to be there now When my blood beams in your head When my feelings are so hollow, hollow"
]

def clean_text(text):
    for phrase in HALLUCINATIONS:
        text = text.replace(phrase, "")
    return text.strip()

# Helper: split audio into chunks with ffmpeg
def split_audio(file_path, chunk_length=60):
    os.makedirs("chunks", exist_ok=True)

    # Copy to a safe temp name in case of special characters
    tmp_dir = tempfile.mkdtemp()
    safe_file = os.path.join(tmp_dir, "temp.wav")
    shutil.copy(file_path, safe_file)

    # Re-encode into clean PCM wav chunks
    cmd = [
        "ffmpeg", "-i", safe_file,
        "-f", "segment", "-segment_time", str(chunk_length),
        "-ar", "16000", "-ac", "1", "chunks/out%03d.wav", "-y"
    ]
    subprocess.run(cmd, check=True)
    return sorted([os.path.join("chunks", f) for f in os.listdir("chunks") if f.endswith(".wav")])

# ===== MAIN LOOP ====
for filename in os.listdir(AUDIO_DIR):
    if not filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
        continue  # skip non-audio files

    audio_path = os.path.join(AUDIO_DIR, filename)
    basename = os.path.splitext(filename)[0]
    raw_file = os.path.join(RAW_DIR, f"{basename}_raw.txt")
    clean_file = os.path.join(CLEAN_DIR, f"{basename}_clean.txt")

    print(f"\n🎵 Processing: {filename}")
    chunks = split_audio(audio_path, CHUNK_LENGTH)

    raw_text = ""
    clean_text_all = ""

    # Transcribe each chunk
    for i, chunk in enumerate(chunks):
        print(f"→ Transcribing chunk {i+1}/{len(chunks)}: {chunk}")
        result = model.transcribe(chunk, fp16=False)

        # Raw file (unformatted)
        raw_text += result["text"].strip() + "\n\n"

        # Clean file (line breaks, hallucinations deleted)
        for seg in result["segments"]:
            clean_line = clean_text(seg["text"]).strip()
            if clean_line:
                clean_text_all += clean_line + "\n"
        clean_text_all += "\n"

    # Save raw output
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Save clean output
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write(clean_text_all)

    print(f"Done: raw → {raw_file}, clean → {clean_file}")