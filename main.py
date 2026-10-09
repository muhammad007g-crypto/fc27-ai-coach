import json
import os
import re
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai

APP_TITLE = "FC 27 AI Coach API"
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "80"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title=APP_TITLE, version="2.0.0")
# For a public deployment, replace "*" with your exact frontend origin.
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
        "message": "POST a gameplay video to /analyze."
    }

@app.get("/health")
def health():
    return {"ok": True, "configured": bool(API_KEY), "model": MODEL}

PROMPT = """
You are a careful EA SPORTS FC 27 gameplay coach. Analyze the actual uploaded gameplay video,
not a generic imagined match. The recording is a football video game, and the user's goal is
to improve their decisions.

Return ONLY valid JSON, no markdown, using exactly this structure:
{
 "overall_score": 0,
 "headline": "short summary",
 "summary": "2-4 sentence evidence-based summary",
 "category_scores": {
   "defending": 0, "attacking": 0, "passing": 0,
   "positioning": 0, "decision_making": 0
 },
 "mistakes": [
   {
     "timestamp": "MM:SS or 'approx. MM:SS'",
     "title": "short mistake/observation",
     "what_happened": "Describe only what is visibly supported by the clip.",
     "why_it_matters": "Explain the gameplay consequence.",
     "better_choice": "Give a specific alternative action.",
     "confidence": "High, Medium, or Low"
   }
 ],
 "improvement_plan": ["3-5 specific practice actions"],
 "coach_focus": "The single most important thing to focus on next match"
}

Rules:
- Scores are coaching estimates, not official game statistics. Never imply measured inputs or button presses
  unless they are visible/audible in the video.
- Do not invent timestamps, player identities, controller inputs, passes, or mistakes.
- If the clip does not provide enough evidence for a category, give a neutral score of 50 and explain the uncertainty.
- Since video sampling can miss fast actions, label timestamps approximate unless the moment is clear.
- Focus on FC gameplay: defensive shape, switching implications only when inferable, jockeying/pressing choices,
  covering passing lanes, controlling centre-backs, spacing, passing options, attack tempo, sprinting, and shot choices.
- Distinguish a definite error from a possible alternative. Include up to 5 meaningful observations, not filler.
- Be constructive and specific. Explain what the user should do instead.
- If the clip mostly shows a replay, set piece, menu, or unclear gameplay, say so and do not fabricate a full analysis.
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
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
    raise ValueError("The AI response was not valid JSON. Please try a shorter clip.")

@app.post("/analyze")
async def analyze(video: UploadFile = File(...), focus: str = "general"):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="The backend has no GEMINI_API_KEY configured yet.")
    filename = video.filename or "gameplay.mp4"
    ext = Path(filename).suffix.lower()
    allowed = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpeg", ".mpg", ".wmv", ".3gp"}
    if ext and ext not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported video extension. Please use MP4, MOV, or WebM.")
    mime = video.content_type or "video/mp4"
    if not mime.startswith("video/"):
        mime = "video/mp4"

    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Video is too large. This server limit is {MAX_FILE_MB} MB; upload a shorter/compressed clip.")

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
        while getattr(uploaded_file, "state", None) and uploaded_file.state.name == "PROCESSING":
            if time.time() > deadline:
                raise HTTPException(status_code=504, detail="Video processing timed out. Try a shorter clip.")
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if getattr(uploaded_file, "state", None) and uploaded_file.state.name == "FAILED":
            raise HTTPException(status_code=422, detail="Google could not process this video. Try MP4 format.")

        models_to_try = [MODEL, "gemini-3.7-flash"]
        response = None
        last_error = None

        for model_name in dict.fromkeys(models_to_try):
            for attempt in range(2):
                try:
                    print(
                        f"Trying AI model: {model_name}, "
                        f"attempt {attempt + 1}/2"
                    )

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


        if response is None:
            raise last_error or RuntimeError("AI analysis failed.")
        result = extract_json(getattr(response, "text", "") or "")
        # Keep output safe and predictable for the frontend.
        result.setdefault("overall_score", 50)
        result.setdefault("headline", "FC 27 coaching report")
        result.setdefault("summary", "Review the observations below.")
        result.setdefault("category_scores", {})
        result.setdefault("mistakes", [])
        result.setdefault("improvement_plan", [])
        result.setdefault("coach_focus", "Focus on the most clearly supported improvement.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        # Do not leak credentials or detailed internals into the public response.
        print(f"Analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail="AI analysis failed. Check the server logs, API key, model access, and usage limits.")
    finally:
        try:
            await video.close()
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        # Uploaded Google files are intentionally not deleted here because SDK delete
        # methods can vary; configure retention/cleanup for production deployments.
