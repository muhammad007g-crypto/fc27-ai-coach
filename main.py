import base64
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai

APP_TITLE = "FC 27 AI Coach API"
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "400"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title=APP_TITLE, version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "status": "ok",
        "model": MODEL,
        "configured": bool(API_KEY),
        "message": "POST short clips to /analyze or longer clips to /analyze-async, then poll /jobs/{job_id}.",
    }


@app.get("/health")
def health():
    return {"ok": True, "configured": bool(API_KEY), "model": MODEL}


PROMPT = """
You are a cautious, evidence-led coach for EA SPORTS FC 27. Analyze ONLY the uploaded gameplay video. Your priority is correctness, not finding as many mistakes as possible.

Return ONLY valid JSON, with exactly this structure:
{
  "overall_score": 0,
  "headline": "short evidence-based summary",
  "summary": "2-4 sentences describing what the clip actually shows and any important uncertainty",
  "category_scores": {
    "defending": 0, "attacking": 0, "passing": 0,
    "positioning": 0, "decision_making": 0
  },
  "mistakes": [
    {
      "timestamp": "approx. MM:SS",
      "title": "short, specific observation",
      "what_happened": "observable action only; distinguish observation from interpretation",
      "why_it_matters": "likely gameplay consequence, without claiming an outcome not shown",
      "better_choice": "one practical alternative the user could try",
      "confidence": "High, Medium, or Low",
      "annotations": [
        {"type": "circle", "x": 500, "y": 500, "r": 38, "label": "watch"},
        {"type": "arrow", "x1": 300, "y1": 600, "x2": 500, "y2": 500, "label": "option"}
      ]
    }
  ],
  "improvement_plan": ["3-5 specific practice actions"],
  "coach_focus": "the single most useful focus for the next match"
}

ACCURACY RULES — FOLLOW THESE STRICTLY:
1. Describe only events you can actually see in this video. Never invent a pass, tackle, turnover, shot, goal, player run, scoreline, match context, or outcome.
2. Do not guess player names. Use a name only if it is clearly readable and relevant; otherwise say "the ball carrier", "a defender", "your midfielder", or similar.
3. Do not claim the user pressed a button, used right-stick switching, sprinted, jockeyed, or selected a specific player unless the recording clearly proves it. Describe visible movement instead.
4. Before calling something a mistake, check whether the relevant action and its consequence are visible. If the evidence is ambiguous, label it as a possible improvement with Medium/Low confidence, or omit it.
5. Do not infer a conceded goal, an unmarked runner, a blocked pass, or a dangerous counter unless the clip actually shows enough evidence. Use wording such as "could leave space" for a risk, not "left space" as a fact when unclear.
6. Timestamps must point to the approximate moment of the described event in the uploaded clip. Do not make up precise timestamps. Use "approx. MM:SS" and keep each observation tied to a moment that is actually shown.
7. Video sampling can miss fast transitions. If you cannot confidently reconstruct the sequence between sampled moments, state that limitation rather than filling in the gaps.
8. Give up to 3 strong, distinct mistakes; fewer is better than fabricated or repetitive criticism. Do not force a mistake if the clip does not support one.
9. Scores are rough coaching estimates based only on the evidence in this clip, not official EA statistics or a definitive rating of the player's skill. Use 50 when there is insufficient evidence for a category.
10. Keep observations separate from tactical advice. Advice may describe a better option, but do not present that option as proof that the original choice was wrong.
11. If the video mainly shows menus, replays, a set piece, distant play, or unclear footage, say the evidence is limited and do not fabricate a full report.
12. Tailor advice to FC gameplay: passing lanes, support options, spacing, defensive depth, pressing, jockeying, switching, attacking choices, and transition risks — but mention a mechanic only when relevant to what is visible.
13. Be concise, specific, and constructive. Do not repeat the same issue in the mistakes, improvement plan, and coach focus using different words.
14. Return valid JSON only. Do not include markdown fences or extra text.
15. For each mistake, provide an "annotations" array with 0-3 visual marks that directly match the frame at that timestamp. Use coordinates normalized to a 0-1000 grid across the full screenshot: x=0 is left, x=1000 is right, y=0 is top, y=1000 is bottom. A circle must use {"type":"circle","x":number,"y":number,"r":number,"label":"short label"}; x/y are the circle center and r is a radius from 18 to 75. An arrow must use {"type":"arrow","x1":number,"y1":number,"x2":number,"y2":number,"label":"short label"}. For arrows, start at x1/y1 and point toward x2/y2. Use circles to identify a clearly visible player or space; use arrows only for a visible or tactically relevant movement/passing option. Do not guess exact positions. If the frame is too unclear, return an empty annotations array. Never add annotations that contradict the screenshot.
"""


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("The AI response was not valid JSON. Please try a shorter clip.")


def timestamp_to_seconds(value: str) -> float | None:
    """Parse MM:SS or HH:MM:SS, including timestamps prefixed with 'approx.'."""
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", value or "")
    if not match:
        return None
    first, second, third = match.groups()
    if third is not None:
        return int(first) * 3600 + int(second) * 60 + int(third)
    return int(first) * 60 + int(second)


def add_gameplay_frames(result: dict, video_path: str) -> None:
    """Attach small JPEG screenshots close to each mistake timestamp; never fail analysis if frame extraction fails."""
    mistakes = result.get("mistakes")
    if not isinstance(mistakes, list) or not mistakes:
        return

    capture = None
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print("Gameplay screenshots unavailable: OpenCV could not open the uploaded video.")
            return

        for mistake in mistakes[:5]:
            if not isinstance(mistake, dict):
                continue
            seconds = timestamp_to_seconds(str(mistake.get("timestamp", "")))
            if seconds is None or seconds < 0:
                continue

            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                # Try a moment shortly before the reported time if the exact frame is unavailable.
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0, seconds - 0.5) * 1000)
                ok, frame = capture.read()
            if not ok or frame is None:
                continue

            height, width = frame.shape[:2]
            max_width = 640
            if width > max_width:
                new_height = max(1, int(height * max_width / width))
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)

            encoded_ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            )
            if not encoded_ok:
                continue

            image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
            mistake["screenshot_data_url"] = "data:image/jpeg;base64," + image_b64
    except Exception as exc:
        print(f"Gameplay screenshot extraction skipped: {type(exc).__name__}: {exc}")
    finally:
        if capture is not None:
            capture.release()


# In-memory job storage is suitable for this single-instance MVP. If Render restarts,
# active jobs are lost; persistent job storage would be needed for production.
JOBS: dict[str, dict] = {}


def save_upload_to_temp(video: UploadFile, ext: str) -> str:
    """Stream upload to disk in chunks instead of holding a long video in RAM."""
    temp_path = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp:
            temp_path = tmp.name
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {MAX_FILE_MB} MB server limit. Compress/export at 720p and try again.",
                    )
                tmp.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")
        return temp_path
    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def analyze_saved_video(temp_path: str, focus: str) -> dict:
    """Upload a saved video to Gemini, wait for processing, analyze, and attach frames."""
    if not API_KEY:
        raise RuntimeError("The backend has no GEMINI_API_KEY configured.")

    client = genai.Client(api_key=API_KEY)
    uploaded_file = client.files.upload(file=temp_path)

    # Long videos may take several minutes for Gemini to finish processing.
    deadline = time.time() + 900
    while (
        getattr(uploaded_file, "state", None)
        and uploaded_file.state.name == "PROCESSING"
    ):
        if time.time() > deadline:
            raise RuntimeError("Google video processing exceeded 15 minutes. Try a compressed 720p clip.")
        time.sleep(3)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if getattr(uploaded_file, "state", None) and uploaded_file.state.name == "FAILED":
        raise RuntimeError("Google could not process this video. Try MP4 format or export it at 720p.")

    models_to_try = [MODEL, "gemini-3.7-flash"]
    response = None
    last_error = None
    for model_name in dict.fromkeys(models_to_try):
        for attempt in range(2):
            try:
                print(f"Trying AI model: {model_name}, attempt {attempt + 1}/2")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        uploaded_file,
                        PROMPT + "\nAnalyze the full uploaded video chronologically, including the beginning, middle, and end. Do not treat the clip as if it were only a short excerpt. "
                        + f"\nUser-selected coaching focus: {focus}. If general, cover the most important decisions.",
                    ],
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                temporary_error = any(marker in error_text for marker in [
                    "429", "500", "502", "503", "504", "unavailable",
                    "resource_exhausted", "internal error", "timeout", "timed out",
                    "connection", "temporarily",
                ])
                if not temporary_error:
                    break
                if attempt == 0:
                    time.sleep(3)
        if response is not None:
            break
        if last_error is not None:
            error_text = str(last_error).lower()
            temporary_error = any(marker in error_text for marker in [
                "429", "500", "502", "503", "504", "unavailable",
                "resource_exhausted", "internal error", "timeout", "timed out",
                "connection", "temporarily",
            ])
            if not temporary_error:
                raise last_error

    if response is None:
        raise last_error or RuntimeError("AI analysis failed.")

    result = extract_json(getattr(response, "text", "") or "")
    result.setdefault("overall_score", 50)
    result.setdefault("headline", "FC 27 coaching report")
    result.setdefault("summary", "Review the observations below.")
    result.setdefault("category_scores", {})
    result.setdefault("mistakes", [])
    result.setdefault("improvement_plan", [])
    result.setdefault("coach_focus", "Focus on the most clearly supported improvement.")
    add_gameplay_frames(result, temp_path)
    return result


def run_analysis_job(job_id: str, temp_path: str, focus: str) -> None:
    try:
        JOBS[job_id] = {"status": "processing", "progress_message": "Gemini is analyzing the full video."}
        result = analyze_saved_video(temp_path, focus)
        JOBS[job_id] = {"status": "completed", "result": result}
    except Exception as exc:
        print(f"Long-video job {job_id} failed: {type(exc).__name__}: {exc}")
        JOBS[job_id] = {"status": "failed", "detail": str(exc) or "AI analysis failed."}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def validate_upload(video: UploadFile) -> str:
    filename = video.filename or "gameplay.mp4"
    ext = Path(filename).suffix.lower()
    allowed = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpeg", ".mpg", ".wmv", ".3gp"}
    if ext and ext not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported video extension. Please use MP4, MOV, or WebM.")
    if not API_KEY:
        raise HTTPException(status_code=503, detail="The backend has no GEMINI_API_KEY configured yet.")
    return ext or ".mp4"


@app.post("/analyze-async")
async def analyze_async(background_tasks: BackgroundTasks, video: UploadFile = File(...), focus: str = "general"):
    """Upload a clip and return immediately; poll /jobs/{job_id} for the result."""
    ext = validate_upload(video)
    temp_path = None
    try:
        temp_path = save_upload_to_temp(video, ext)
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"status": "queued", "progress_message": "Upload received; waiting to analyze."}
        background_tasks.add_task(run_analysis_job, job_id, temp_path, focus)
        temp_path = None  # the background worker now owns cleanup
        return {"job_id": job_id, "status": "queued", "message": "Upload complete. Analysis continues in the background."}
    finally:
        await video.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found. The server may have restarted; please upload again.")
    return job


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), focus: str = "general"):
    """Compatibility endpoint for short clips; long clips should use /analyze-async."""
    ext = validate_upload(video)
    temp_path = None
    try:
        temp_path = save_upload_to_temp(video, ext)
        return analyze_saved_video(temp_path, focus)
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="AI analysis failed. Check Render logs, Gemini API configuration, model access, and usage limits. For long clips, use the background analysis flow.")
    finally:
        await video.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

