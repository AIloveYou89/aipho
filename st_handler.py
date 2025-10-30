# PhoWhisper (VinAI) on faster-whisper / CTranslate2 — Runpod Serverless
# st_handler.py — patched to reduce end-of-file repeats and improve VAD chunking
# Updated: 2025-10-30

import os, uuid, time, json, logging, tempfile, subprocess, math, shutil
from typing import Dict, Any, List, Optional, Tuple

import runpod
import requests
from faster_whisper import WhisperModel

# =========================
# ENV (override khi deploy)
# =========================
MODEL_ID     = os.getenv("MODEL_ID", "kiendt/PhoWhisper-large-ct2")  # HF repo CT2 đã convert
MODEL_DIR    = os.getenv("MODEL_DIR", "/models")                     # nơi cache model CT2
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/out")

DEVICE       = os.getenv("DEVICE", "cuda")                           # cuda | cpu
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")                  # float16 | int8_float16 | int8
LANG         = os.getenv("LANG", "vi")                               # "vi" hoặc "auto"
VAD_FILTER   = os.getenv("VAD_FILTER", "1") == "1"                   # cắt lặng
MAX_CHUNK    = float(os.getenv("MAX_CHUNK_LEN", "30"))               # giây, cho long-form

# Patch chống lặp/đứt cuối câu
COND_PREV    = os.getenv("COND_PREV", "1") == "1"                    # condition_on_previous_text
NO_SPEECH_TH = float(os.getenv("NO_SPEECH_THRESHOLD", "0.7"))        # bỏ near-silence
VAD_MIN_SIL  = int(os.getenv("VAD_MIN_SIL_MS", "600"))               # ms im lặng để cắt

MAKE_SRT     = os.getenv("SRT", "1") == "1"
MAKE_VTT     = os.getenv("VTT", "0") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PhoWhisper-Serverless")

# =========================
# Lazy model loader
# =========================
_model: Optional[WhisperModel] = None
def get_model() -> WhisperModel:
    global _model
    if _model is None:
        log.info(f"[MODEL] Loading: {MODEL_ID} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
        _model = WhisperModel(
            MODEL_ID,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=MODEL_DIR
        )
        log.info("[MODEL] Loaded successfully.")
    return _model

# =========================
# Utils
# =========================
def _download(url: str, to_path: str, timeout: int = 300) -> str:
    log.info(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return to_path

def _to_wav_16k_mono(src_path: str, dst_path: str) -> str:
    """Chuẩn hoá audio về 16kHz mono s16 bằng ffmpeg"""
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", dst_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {res.stderr[-500:]}")
    return dst_path

def _fmt_ts(sec: float) -> str:
    if sec < 0: sec = 0.0
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    ms = int((sec - math.floor(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _write_srt(segments: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def _write_vtt(segments: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            s = _fmt_ts(seg["start"]).replace(",", ".")
            e = _fmt_ts(seg["end"]).replace(",", ".")
            f.write(f"{s} --> {e}\n{seg['text'].strip()}\n\n")

def _normalize_input(inp: Dict[str, Any]) -> Tuple[str, str]:
    """
    Hỗ trợ:
      - audio_path: đường dẫn local (đã mount volume)
      - audio_url : link http(s) để tải về rồi xử lý
    Trả về: (wav_16k_path, tmp_dir)
    """
    audio_path = inp.get("audio_path")
    audio_url  = inp.get("audio_url")
    if not audio_path and not audio_url:
        raise ValueError("Provide 'audio_path' or 'audio_url'.")

    tmp_dir = tempfile.mkdtemp(prefix="pho_")
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio_path not found: {audio_path}")
        raw = audio_path
    else:
        raw = os.path.join(tmp_dir, "raw.input")
        _download(audio_url, raw)

    fixed = os.path.join(tmp_dir, "fixed_16k.wav")
    _to_wav_16k_mono(raw, fixed)
    return fixed, tmp_dir

def strip_overlap(prev_text: str, curr_text: str, max_overlap: int = 24) -> str:
    """Gọt phần trùng giữa đuôi đoạn trước và đầu đoạn hiện tại (chống lặp)."""
    prev_tail = prev_text[-max_overlap:].lower()
    t = curr_text.strip()
    # thử từ dài -> ngắn để ưu tiên khớp dài
    for m in range(min(len(t), max_overlap), 6, -1):
        if prev_tail.endswith(t[:m].lower()):
            return t[m:].lstrip()
    return t

# =========================
# Handler
# =========================
def handler(event):
    """
    Input JSON:
    {
      "audio_path": "/runpod-volume/audio/test.wav",  // hoặc
      "audio_url": "https://.../sample.mp3",
      "beam_size": 5,
      "temperature": 0.2,
      "word_timestamps": true,
      "return": "json|text",
      "outfile_prefix": "pho_demo"
    }
    """
    t0 = time.time()
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    tmp_dir = None

    try:
        wav16, tmp_dir = _normalize_input(inp)
    except Exception as e:
        log.error(f"[INPUT] {e}")
        return {"error": f"Input error: {e}"}

    beam_size   = int(inp.get("beam_size", 5))
    temperature = float(inp.get("temperature", 0.2))
    word_ts     = bool(inp.get("word_timestamps", True))
    lang = None if str(LANG).lower() == "auto" else LANG

    job_id  = str(uuid.uuid4())
    prefix  = inp.get("outfile_prefix") or job_id
    json_p  = os.path.join(OUT_DIR, f"{prefix}.json")
    srt_p   = os.path.join(OUT_DIR, f"{prefix}.srt")
    vtt_p   = os.path.join(OUT_DIR, f"{prefix}.vtt")

    try:
        model = get_model()
        log.info(f"[ASR] job_id={job_id} | beam={beam_size} | chunk={MAX_CHUNK}s | vad={VAD_FILTER}")
        segments_gen, info = model.transcribe(
            wav16,
            language=lang,
            task="transcribe",
            vad_filter=VAD_FILTER,
            vad_parameters={"min_silence_duration_ms": VAD_MIN_SIL},
            beam_size=beam_size,
            temperature=temperature,
            word_timestamps=word_ts,
            chunk_length=MAX_CHUNK,
            condition_on_previous_text=COND_PREV,
            no_speech_threshold=NO_SPEECH_TH,
            compression_ratio_threshold=2.6,
            log_prob_threshold=-0.5,
            suppress_blank=True,
            suppress_non_speech_tokens=True
        )
    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        log.error(f"[ASR] {e}")
        return {"error": f"Transcribe failed: {e}"}

    segments: List[Dict[str, Any]] = []
    words_all: List[Dict[str, Any]] = []
    texts: List[str] = []

    for seg in segments_gen:
        item = {
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        }
        if word_ts and seg.words:
            item["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
            words_all.extend(item["words"])

        # Chống lặp giữa các segment (đặc biệt ở ranh giới chunk)
        joined = " ".join(texts)
        clean = strip_overlap(joined, item["text"])
        item["text"] = clean

        segments.append(item)
        texts.append(clean)

    elapsed = round(time.time() - t0, 2)
    log.info(f"[DONE] {len(segments)} segments | {elapsed}s")

    result = {
        "job_id": job_id,
        "model_id": MODEL_ID,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "elapsed_sec": elapsed,
        "num_segments": len(segments),
        "num_words": len(words_all),
        "text": " ".join(texts).strip(),
        "segments": segments,
        "outputs": {"json": json_p, "srt": None, "vtt": None}
    }

    # Lưu JSON + phụ đề (tuỳ chọn)
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    try:
        if MAKE_SRT:
            _write_srt(segments, srt_p)
            result["outputs"]["srt"] = srt_p
        if MAKE_VTT:
            _write_vtt(segments, vtt_p)
            result["outputs"]["vtt"] = vtt_p
    except Exception as e:
        log.warning(f"[SUBTITLE] {e}")

    # Cleanup
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            log.warning(f"[CLEANUP] {e}")

    # Kiểu trả về
    if inp.get("return", "json") == "text":
        return {
            "job_id": job_id,
            "elapsed_sec": result["elapsed_sec"],
            "language": result["language"],
            "path_json": json_p,
            "path_srt": result["outputs"]["srt"],
            "text": result["text"]
        }
    return result

# =========================
# Runpod serverless entry
# =========================
log.info("[INIT] Starting Runpod serverless worker…")
runpod.serverless.start({"handler": handler})
