# PhoWhisper (VinAI) on faster-whisper / CTranslate2 – Runpod Serverless
# st_handler.py - Auto-split subtitles + Fixed hallucination
# Updated: 2025-10-30 - SRT OUTPUT ONLY VERSION

import os, uuid, time, logging, tempfile, subprocess, math, shutil
from typing import Dict, Any, List, Optional, Tuple

import runpod
import requests
from faster_whisper import WhisperModel

# -----------------------------
# ENV
# -----------------------------
MODEL_ID     = os.getenv("MODEL_ID", "kiendt/PhoWhisper-large-ct2")
MODEL_DIR    = os.getenv("MODEL_DIR", "/models")
OUT_DIR      = os.getenv("OUT_DIR", "/runpod-volume/out")

DEVICE       = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
LANG         = os.getenv("LANG", "vi")
VAD_FILTER   = os.getenv("VAD_FILTER", "1") == "1"
MAX_CHUNK    = float(os.getenv("MAX_CHUNK_LEN", "30"))

# Chống hallucination
NO_SPEECH_TH = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))
LOGPROB_TH   = float(os.getenv("LOGPROB_THRESHOLD", "-1.0"))
COMPRESSION_TH = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "2.4"))
MIN_SEGMENT_DURATION = float(os.getenv("MIN_SEGMENT_DURATION", "0.3"))

# ✨ THAM SỐ CHIA SUBTITLE
MAX_CHARS_PER_LINE = int(os.getenv("MAX_CHARS_PER_LINE", "28"))
MAX_DURATION_PER_LINE = float(os.getenv("MAX_DURATION_PER_LINE", "2"))

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PhoWhisper-Serverless")

# -----------------------------
# Lazy model loader
# -----------------------------
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

# -----------------------------
# Utils
# -----------------------------
def _download(url: str, to_path: str, timeout: int = 180) -> str:
    log.info(f"[DOWNLOAD] {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(to_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return to_path

def _to_wav_16k_mono(src_path: str, dst_path: str) -> str:
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

def _normalize_input(inp: Dict[str, Any]) -> Tuple[str, str]:
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

def _is_valid_segment(seg, prev_text: str = "") -> bool:
    """Lọc bỏ segments không hợp lệ"""
    duration = seg.end - seg.start
    text = seg.text.strip()
    
    if duration < MIN_SEGMENT_DURATION:
        log.debug(f"[FILTER] Skipped short segment ({duration:.3f}s): {text[:30]}")
        return False
    
    if not text or len(text) < 3:
        log.debug(f"[FILTER] Skipped empty/short text: '{text}'")
        return False
    
    if prev_text and text in prev_text:
        log.debug(f"[FILTER] Skipped repetition: {text[:30]}")
        return False
    
    return True

# ✨ HÀM CHIA SEGMENTS THÀNH DÒNG NGẮN
def _split_segment_by_words(seg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chia 1 segment dài thành nhiều dòng ngắn dựa trên:
    - Số ký tự (MAX_CHARS_PER_LINE)
    - Duration (MAX_DURATION_PER_LINE, mặc định ~2s)
    - Word timestamps
    - Sửa lỗi dính chữ (thêm khoảng trắng phù hợp)
    """
    words = seg.get("words", [])
    if not words:
        return [seg]
    
    result_segments = []
    current_words: List[Dict[str, Any]] = []
    current_text = ""
    start_time = words[0]["start"]
    
    no_space_prefix = set(",.!?;:…)]}»""'")

    for i, word in enumerate(words):
        word_text = word["word"]
        sep = "" if (not current_text or (word_text and word_text[0] in no_space_prefix)) else " "
        test_text = current_text + sep + word_text
        test_duration = word["end"] - start_time

        should_break = (
            len(test_text) > MAX_CHARS_PER_LINE or 
            test_duration > MAX_DURATION_PER_LINE
        )
        
        if should_break and current_words:
            result_segments.append({
                "start": start_time,
                "end": current_words[-1]["end"],
                "text": current_text.strip()
            })
            current_words = [word]
            current_text = word_text
            start_time = word["start"]
        else:
            current_words.append(word)
            current_text = test_text
    
    if current_words:
        result_segments.append({
            "start": start_time,
            "end": current_words[-1]["end"],
            "text": current_text.strip()
        })
    
    return result_segments

# -----------------------------
# Handler - CHỈ TRẢ VỀ FILE SRT
# -----------------------------
def handler(event):
    t0 = time.time()
    inp = event.get("input", {}) if isinstance(event, dict) else {}
    tmp_dir = None

    try:
        wav16, tmp_dir = _normalize_input(inp)
    except Exception as e:
        log.error(f"[INPUT] {e}")
        return {"error": f"Input error: {e}"}

    beam_size   = int(inp.get("beam_size", 5))
    temperature = float(inp.get("temperature", 0.0))
    word_ts     = True
    lang = None if str(LANG).lower() == "auto" else LANG

    job_id  = str(uuid.uuid4())
    prefix  = inp.get("outfile_prefix") or job_id
    srt_p   = os.path.join(OUT_DIR, f"{prefix}.srt")

    try:
        model = get_model()
        log.info(f"[ASR] job_id={job_id} | beam={beam_size} | chunk={MAX_CHUNK}s | vad={VAD_FILTER}")
        segments_gen, info = model.transcribe(
            wav16,
            language=lang,
            task="transcribe",
            vad_filter=VAD_FILTER,
            beam_size=beam_size,
            temperature=temperature,
            word_timestamps=word_ts,
            chunk_length=MAX_CHUNK,
            condition_on_previous_text=False,
            no_speech_threshold=NO_SPEECH_TH,
            log_prob_threshold=LOGPROB_TH,
            compression_ratio_threshold=COMPRESSION_TH
        )
    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        log.error(f"[ASR] {e}")
        return {"error": f"Transcribe failed: {e}"}

    raw_segments: List[Dict[str, Any]] = []
    prev_text = ""

    for seg in segments_gen:
        if not _is_valid_segment(seg, prev_text):
            continue
        
        item = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": []
        }
        
        if seg.words:
            item["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
        
        raw_segments.append(item)
        prev_text = " ".join([s["text"] for s in raw_segments[-3:]])

    # ✨ CHIA SEGMENTS THÀNH DÒNG NGẮN
    final_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        split_segs = _split_segment_by_words(seg)
        final_segments.extend(split_segs)
    
    log.info(f"[SPLIT] {len(raw_segments)} raw segments → {len(final_segments)} final lines")

    elapsed = round(time.time() - t0, 2)
    log.info(f"[DONE] {len(final_segments)} lines | {elapsed}s")

    # GHI FILE SRT
    try:
        _write_srt(final_segments, srt_p)
    except Exception as e:
        log.error(f"[SUBTITLE] {e}")
        return {"error": f"Failed to write SRT: {e}"}

    # CLEANUP
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            log.warning(f"[CLEANUP] {e}")

    # ✅ CHỈ TRẢ VỀ ĐƯỜNG DẪN FILE SRT
    return {
        "srt_path": srt_p,
        "elapsed_sec": elapsed
    }

# -----------------------------
# Runpod serverless entrypoint
# -----------------------------
log.info("[INIT] Starting Runpod serverless worker…")
runpod.serverless.start({"handler": handler})
