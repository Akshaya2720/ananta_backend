# app.py
import os
import logging
from flask import Flask, render_template, request, jsonify

# Optional imports (we'll only use if configured)
_openai_available = False
_gemini_available = False
try:
    import openai
    _openai_available = True
except Exception:
    _openai_available = False

try:
    from google import genai
    from google.genai import types as genai_types
    _gemini_available = True
except Exception:
    _gemini_available = False

# Load dotenv if present (recommended for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__, template_folder="templates")
logging.basicConfig(level=logging.INFO)

# Choose backend: "simple" (default), "openai", "gemini"
BACKEND = os.getenv("CHATBOT_BACKEND", "simple").lower()

# Initialize OpenAI client if requested and key exists
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
if BACKEND == "openai":
    if not _openai_available:
        logging.warning("OpenAI library not installed; OpenAI backend won't work unless you install `openai`.")
        BACKEND = "simple"
    elif not OPENAI_API_KEY:
        logging.warning("OPENAI_API_KEY not found; falling back to simple backend.")
        BACKEND = "simple"
    else:
        openai.api_key = OPENAI_API_KEY
        logging.info("OpenAI backend active. Model: %s", OPENAI_MODEL)

# Initialize Google GenAI (Gemini) client if requested and key exists
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if BACKEND == "gemini":
    if not _gemini_available:
        logging.warning("google-genai not installed; Gemini backend won't work unless you install `google-genai`.")
        BACKEND = "simple"
    elif not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not found; falling back to simple backend.")
        BACKEND = "simple"
    else:
        # ensure env var is set for SDK
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
        except TypeError:
            genai_client = genai.Client()
        logging.info("Gemini backend active. Model: %s", GEMINI_MODEL)

logging.info("Chatbot backend selected: %s", BACKEND)

# -------------------- page routes --------------------
@app.route("/")
def home():
    return render_template("dashboard.html")  # default page

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/labs")
def labs():
    return render_template("labs.html")

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/login")
def login():
    return render_template("login.html")

# -------------------- helper: get bot reply --------------------
def get_bot_reply_simple(user_message: str) -> str:
    """Very small rule-based fallback bot."""
    text = user_message.lower()
    if "hello" in text or "hi" in text:
        return "Hi there! How can I help you today?"
    if "course" in text or "courses" in text:
        return "You can find your courses under the Courses section on the dashboard."
    if "assignment" in text:
        return "Which subject is the assignment for? I can help with tips and resources."
    if "help" in text:
        return "Tell me what you need help with — study plan, assignments, or resources?"
    return "I'm still learning — can you rephrase that or ask about courses, assignments or login?"

def get_bot_reply_openai(user_message: str) -> str:
    """Call OpenAI chat completions. Requires OPENAI_API_KEY & openai package."""
    system_prompt = (
        "You are Ananta, an educational assistant for students. Be concise, helpful, and friendly."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=250,
            temperature=0.2,
            n=1,
        )
        # standard format
        choice = resp["choices"][0]
        return choice["message"]["content"].strip()
    except Exception as e:
        logging.exception("OpenAI call failed")
        raise

def get_bot_reply_gemini(user_message: str) -> str:
    """Call Google Gemini via google-genai. Requires GEMINI_API_KEY & google-genai package."""
    prompt = f"User: {user_message}\nAssistant:"
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=300,
                temperature=0.2,
                candidate_count=1
            ),
        )
        # try response.text first
        if hasattr(response, "text") and response.text:
            return response.text
        # fallback to output structure
        outputs = getattr(response, "output", None)
        if outputs and isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            first = outputs[0]
            if isinstance(first, dict):
                content = first.get("content", None)
                if isinstance(content, list) and len(content) > 0:
                    item = content[0]
                    if isinstance(item, dict) and "text" in item:
                        return item["text"]
        # final fallback
        return str(response)
    except Exception:
        logging.exception("Gemini call failed")
        raise

# -------------------- Chat endpoint --------------------
@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        if BACKEND == "openai":
            reply = get_bot_reply_openai(user_message)
        elif BACKEND == "gemini":
            reply = get_bot_reply_gemini(user_message)
        else:
            reply = get_bot_reply_simple(user_message)

        return jsonify({"reply": reply})
    except Exception as e:
        # For dev: return the error message (do not expose stack in prod)
        logging.exception("Error while generating reply")
        return jsonify({"error": str(e)}), 500

# -------------------- run --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
