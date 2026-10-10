import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai

APP_TITLE = "FC 27 AI Coach API"
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "80"))
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
        "message": "POST a gameplay video to /analyze.",
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
      "confidence": "High, Medium, or Low"
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
            max_width = 1280
            if width > max_width:
                new_height = max(1, int(height * max_width / width))
                frame = cv2.resize(frame, (max_width, new_height), interpolation=cv2.INTER_AREA)

            encoded_ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
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


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), focus: str = "general"):
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="The backend has no GEMINI_API_KEY configured yet.",
        )

    filename = video.filename or "gameplay.mp4"
    ext = Path(filename).suffix.lower()
    allowed = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpeg", ".mpg", ".wmv", ".3gp"}
    if ext and ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail="Unsupported video extension. Please use MP4, MOV, or WebM.",
        )

    mime = video.content_type or "video/mp4"
    if not mime.startswith("video/"):
        mime = "video/mp4"

    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Video is too large. This server limit is {MAX_FILE_MB} MB; upload a shorter/compressed clip.",
        )

    temp_path = None
    uploaded_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".mp4") as tmp:
            tmp.write(data)
            temp_path = tmp.name

        client = genai.Client(api_key=API_KEY)
        uploaded_file = client.files.upload(file=temp_path)

        # Wait for Google's file processing to finish.
        deadline = time.time() + 180
        while (
            getattr(uploaded_file, "state", None)
            and uploaded_file.state.name == "PROCESSING"
        ):
            if time.time() > deadline:
                raise HTTPException(
                    status_code=504,
                    detail="Video processing timed out. Try a shorter clip.",
                )
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if (
            getattr(uploaded_file, "state", None)
            and uploaded_file.state.name == "FAILED"
        ):
            raise HTTPException(
                status_code=422,
                detail="Google could not process this video. Try MP4 format.",
            )

        # Retry temporary service failures, then try the configured fallback model.
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
                            PROMPT
                            + f"\nUser-selected coaching focus: {focus}. "
                            "If general, cover the most important decisions.",
                        ],
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    error_text = str(exc).lower()
                    temporary_error = any(
                        marker in error_text
                        for marker in [
                            "429",
                            "500",
                            "502",
                            "503",
                            "504",
                            "unavailable",
                            "resource_exhausted",
                            "internal error",
                            "timeout",
                            "timed out",
                            "connection",
                            "temporarily",
                        ]
                    )
                    if not temporary_error:
                        break
                    if attempt == 0:
                        time.sleep(2)

            if response is not None:
                break

            if last_error is not None:
                error_text = str(last_error).lower()
                temporary_error = any(
                    marker in error_text
                    for marker in [
                        "429", "500", "502", "503", "504", "unavailable",
                        "resource_exhausted", "internal error", "timeout",
                        "timed out", "connection", "temporarily",
                    ]
                )
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

        # Add a screenshot near each timestamp for the HTML page to display.
        # Screenshot extraction is optional; it will not invalidate the coaching report.
        add_gameplay_frames(result, temp_path)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed. Check the server logs, API key, model access, and usage limits.",
        )
    finally:
        try:
            await video.close()
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        # Uploaded Google files are intentionally not deleted here because SDK delete
        # methods can vary; configure retention/cleanup for production deployments.
