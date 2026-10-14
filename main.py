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

You are a cautious, evidence-based EA SPORTS FC 27 gameplay coach. Your highest priority is factual accuracy, not finding as many mistakes as possible.

Analyze only the uploaded gameplay video. Use the visual evidence available. Never invent events, player actions, passing options, button inputs, outcomes, or tactical situations.

Return ONLY valid JSON with exactly this structure:
{
  "overall_score": 0,
  "headline": "short evidence-based summary",
  "summary": "2-4 sentences describing what is clearly visible and any important uncertainty",
  "category_scores": {
    "defending": 0,
    "attacking": 0,
    "passing": 0,
    "positioning": 0,
    "decision_making": 0
  },
  "mistakes": [
    {
      "timestamp": "approx. MM:SS",
      "title": "short, specific observation",
      "what_happened": "describe only the visible action and separate observation from interpretation",
      "why_it_matters": "explain the likely tactical consequence without claiming an unseen outcome",
      "better_choice": "one practical alternative supported by the visible situation",
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

STRICT ACCURACY RULES:
1. Report only events supported by visible video evidence. Never invent passes, tackles, turnovers, interceptions, shots, or player movements.
2. Do not guess player names. Use a name only when clearly readable and relevant; otherwise describe the player by position or location.
3. Do not claim the user pressed a button, used right-stick switching, sprinted, jockeyed, or selected a player unless the input is directly visible. Describe only observable movement.
4. Before calling an action a mistake, assess the visible alternatives, teammates, opponents, available space, and likely risk. Do not label a reasonable decision as a mistake merely because the attack failed.
5. Do not infer a goal, missed opportunity, blocked pass, dangerous counterattack, or lost possession unless the video supports it.
6. Give the timestamp of the event in the uploaded clip, not the in-game match clock. Use the nearest defensible timestamp and never invent precision.
7. Analyze the video chronologically. Consider the beginning, middle, and end. Do not treat a few isolated frames as proof of an entire sequence.
8. Video sampling may miss fast transitions. If the sequence cannot be reconstructed confidently, omit the proposed mistake or explicitly state the uncertainty in the summary.
9. Report at most 3 strong, distinct mistakes. Return fewer mistakes, including none, rather than fabricate or repeat criticism.
10. Confidence must reflect evidence strength. High means the relevant action and context are clearly visible. Medium means the interpretation is plausible but some context is missing. Low means substantial uncertainty; normally omit low-confidence mistakes.
11. Keep observation, tactical interpretation, and advice separate. Advice must be framed as a suggested alternative, not a guaranteed better outcome.
12. Tailor advice to FC gameplay: passing lanes, support options, spacing, defensive depth, pressing, player switching discipline, transition defence, and attacking patience, where supported by evidence.
13. Scores are approximate coaching estimates based only on the uploaded clip, not official EA statistics. Do not assign extreme scores without sufficient evidence.
14. Annotations must point to the relevant visible players, ball, space, or passing lane. Use coordinates from 0 to 1000 relative to the displayed frame. If the correct location is uncertain, use an empty annotations array rather than guessing.
15. Keep recommendations concise, specific, constructive, and non-repetitive. Do not repeat the same criticism across mistakes, improvement_plan, and coach_focus.
16. If the video mainly shows menus, replays, distant play, or unclear footage, say so in the summary and avoid unsupported claims.
17. Return valid JSON only. Do not include Markdown fences or additional text.
18. Analyze only what is actually available in the supplied video. Never claim to have watched frames or sequences that were not provided to the model.
19. Consistency matters: apply these same evidence standards every time. Prefer a stable, well-supported conclusion over a dramatic or overly detailed analysis.
20. Before reporting a mistake, verify that the visible sequence supports both the claimed action and its tactical consequence. If either is uncertain, lower confidence to Medium or Low, describe only what is visible, and do not present speculation as fact. Prefer the same clearly evidenced event over inventing a different explanation on repeated analyses.
21. Every mistake must include an annotations array containing zero to three visual marks that directly match the event. Use an empty array when appropriate.
ANALYZE THESE 12 GAMEPLAY CATEGORIES:

1. Bad player switching: identify only when the visible sequence supports a poor switch or its consequence.
2. Pulling defenders out of position: look for defenders abandoning important spaces or marking responsibilities.
3. Pressing at the wrong time: identify poorly timed pressure that visibly opens space or passing lanes.
4. Giving opponents too much space: identify opponents receiving or advancing with avoidable freedom.
5. Sprinting when it is unhelpful: assess visible movement and its consequences; never claim a sprint button was pressed unless directly observable.
6. Passing to the wrong player: identify a pass when a clearly better, visible option was available.
7. Missing obvious passing options: identify clearly visible, practical passing options that were overlooked.
8. Holding the ball too long: identify avoidable delays that visibly allow pressure or close passing options.
9. Rushing attacks: identify attacks forced forward despite clearly visible safer or more effective options.
10. Poor defensive positioning: identify positioning that visibly weakens defensive coverage.
11. Creating unnecessary gaps: identify avoidable spaces between defenders or midfielders that opponents can exploit.
12. Predictable attacking patterns: identify repeated or easily anticipated attacking choices only when the video provides sufficient evidence.

Evaluate all 12 categories internally for every analysis. Report only the strongest, clearly supported mistakes allowed by the existing reporting rules. Do not force a mistake into every category. If a category cannot be reliably assessed from the video, treat it as unconfirmed rather than guessing. Keep the existing JSON structure and all other instructions unchanged.
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
            max_width = 1920
            if width > max_width:
                new_height = max(1, int(height * max_width / width))
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)

            encoded_ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
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

