# PhoWhisper (VinAI) on faster-whisper / CTranslate2 – Runpod Serverless
# Updated: 2025-10-30
import os, uuid, time, json, logging, tempfile, subprocess, math, shutil
from typing import Dict, Any, List, Optional
import runpod
import requests
from faster_whisper import WhisperModel

# -----------------------------
# ENV
# -----------------------------
MODEL_DIR    = os.getenv("MODEL_DIR", "/models/PhoWhisper-large-ct2")
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/out")
DEVICE       = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
LANG         = os.getenv("LANG", "vi")
VAD_FILTER   = os.getenv("VAD_FILTER", "1") == "1"
MAX_CHUNK    = float(os.getenv("MAX_CHUNK_LEN", "30"))
MAKE_SRT     = os.getenv("SRT", "1") == "1"
MAKE_VTT     = os.getenv("VTT", "0") == "1"

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PhoWhisper-Serverless")

# Lazy model
_model: Optional[WhisperModel] = None
def get_model() -> WhisperModel:
    global _model
    if _model is None:
        log.info(f"[MODEL] Loading from: {MODEL_DIR} (device={DEVICE}, compute_type={COMPUTE_TYPE})")
        _model = WhisperModel(
            MODEL_DIR,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        log.info("[MODEL] Loaded successfully!")
    return _model

# -----------------------------
# Utilities
# -----------------------------
def _download(url: str, to_path: str, timeout: int = 180):
    log.info(f"[DOWNLOAD] Fetching: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    log.info(f"[DOWNLOAD] Saved to: {to_path}")
    return to_path

def _to_wav_16k_mono(src_path: str, dst_path: str):
    """Convert any audio to 16kHz mono s16 with ffmpeg"""
    log.info(f"[CONVERT] Converting to 16kHz mono: {src_path}")
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        dst_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {res.stderr[-500:]}")
    return dst_path

def _fmt_ts(sec: float) -> str:
    if sec < 0: sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - math.floor(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _write_srt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n{seg['text'].strip()}\n\n")

def _write_vtt(segments: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            s = _fmt_ts(seg["start"]).replace(",", ".")
            e = _fmt_ts(seg["end"]).replace(",", ".")
            f.write(f"{s} --> {e}\n{seg['text'].strip()}\n\n")

def _normalize_input(inp: Dict[str, Any]) -> tuple[str, str]:
    """
    Accept either:
      - audio_path: local path (mounted volume)
      - audio_url:  http(s) to download then transcribe
    Returns: (wav_path, tmp_dir_to_cleanup)
    """
    audio_path = inp.get("audio_path")
    audio_url  = inp.get("audio_url")
    if not audio_path and not audio_url:
        raise ValueError("Provide 'audio_path' or 'audio_url'")

    tmp_dir = tempfile.mkdtemp(prefix="pho_")
    
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio_path not found: {audio_path}")
        raw_path = audio_path
    else:
        raw_path = os.path.join(tmp_dir, "raw.input")
        _download(audio_url, raw_path)

    fixed = os.path.join(tmp_dir, "fixed_16k.wav")
    _to_wav_16k_mono(raw_path, fixed)
    return fixed, tmp_dir

# -----------------------------
# Handler
# -----------------------------
def handler(event):
    """
    Input JSON:
    {
      "audio_path": "/runpod-volume/audio/test.wav",   # hoặc
      "audio_url": "https://.../sample.mp3",
      "beam_size": 5,
      "temperature": 0.0,
      "word_timestamps": true,
      "return": "json|text",
      "outfile_prefix": "result_xyz"
    }
    """
    t0 = time.time()
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    tmp_dir = None
    
    try:
        wav16, tmp_dir = _normalize_input(inp)
    except Exception as e:
        log.error(f"[INPUT ERROR] {e}")
        return {"error": f"Input error: {e}"}

    # Options
    beam_size   = int(inp.get("beam_size", 5))
    temperature = float(inp.get("temperature", 0.0))
    word_ts     = bool(inp.get("word_timestamps", True))
    vad_filter  = bool(inp.get("vad_filter", VAD_FILTER))
    
    # VAD parameters - điều chỉnh để tránh hallucination
    default_vad = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 1000,  # Tăng lên để phát hiện kết thúc tốt hơn
        "speech_pad_ms": 400
    }
    vad_params = inp.get("vad_parameters", default_vad if vad_filter else None)
    
    # Condition on previous text để tránh lặp
    condition_on_previous_text = bool(inp.get("condition_on_previous_text", False))
    
    lang = None if str(LANG).lower() == "auto" else LANG

    # Prepare output paths
    job_id = str(uuid.uuid4())
    prefix = inp.get("outfile_prefix") or job_id
    json_path = os.path.join(OUT_DIR, f"{prefix}.json")
    srt_path  = os.path.join(OUT_DIR, f"{prefix}.srt")
    vtt_path  = os.path.join(OUT_DIR, f"{prefix}.vtt")

    # Load + transcribe
    try:
        model = get_model()
        log.info(f"[TRANSCRIBE] Starting transcription (job_id={job_id})")
        
        # Build transcribe parameters
        transcribe_params = {
            "language": lang,
            "task": "transcribe",
            "vad_filter": vad_filter,
            "beam_size": beam_size,
            "temperature": temperature,
            "word_timestamps": word_ts,
            "chunk_length": MAX_CHUNK,
            "condition_on_previous_text": condition_on_previous_text
        }
        
        # Add VAD parameters if VAD is enabled
        if vad_filter and vad_params:
            transcribe_params["vad_parameters"] = vad_params
        
        segments_gen, info = model.transcribe(wav16, **transcribe_params)
    except Exception as e:
        log.error(f"[TRANSCRIBE ERROR] {e}")
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"error": f"Transcribe failed: {e}"}

    # Collect segments with deduplication
    segments = []
    words_all = []
    texts = []
    seen_texts = set()
    
    for seg in segments_gen:
        text = seg.text.strip()
        
        # Skip empty or duplicate segments
        if not text or text in seen_texts:
            log.warning(f"[SKIP] Duplicate/empty segment: '{text}'")
            continue
            
        item = {
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text
        }
        if word_ts and seg.words:
            item["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in seg.words
            ]
            words_all.extend(item["words"])
        
        segments.append(item)
        texts.append(text)
        seen_texts.add(text)

    elapsed = round(time.time() - t0, 2)
    log.info(f"[DONE] Transcribed {len(segments)} segments in {elapsed}s")

    result = {
        "job_id": job_id,
        "model": os.path.basename(MODEL_DIR),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "elapsed_sec": elapsed,
        "num_segments": len(segments),
        "num_words": len(words_all),
        "text": " ".join(texts).strip(),
        "segments": segments,
        "outputs": {"json": json_path, "srt": None, "vtt": None}
    }

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Optional subtitles
    try:
        if MAKE_SRT:
            _write_srt(segments, srt_path)
            result["outputs"]["srt"] = srt_path
            log.info(f"[SRT] Saved: {srt_path}")
        if MAKE_VTT:
            _write_vtt(segments, vtt_path)
            result["outputs"]["vtt"] = vtt_path
            log.info(f"[VTT] Saved: {vtt_path}")
    except Exception as e:
        log.warning(f"[SUBTITLE ERROR] {e}")

    # Cleanup temp directory
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
            log.info(f"[CLEANUP] Removed temp dir: {tmp_dir}")
        except Exception as e:
            log.warning(f"[CLEANUP ERROR] {e}")

    # Return mode
    ret_mode = inp.get("return", "json")
    if ret_mode == "text":
        return {
            "job_id": job_id,
            "elapsed_sec": result["elapsed_sec"],
            "language": result["language"],
            "path_json": json_path,
            "path_srt": result["outputs"]["srt"],
            "text": result["text"]
        }
    return result

# Start serverless worker
log.info("[INIT] Starting RunPod serverless worker...")
runpod.serverless.start({"handler": handler})
