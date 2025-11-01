from flask import Flask, redirect, request, render_template, abort, jsonify
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

app = Flask(__name__)

# ============ Logging ============
logger = logging.getLogger("flask_app")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = RotatingFileHandler(f"website_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

# ============ Config ============
EXPECTED_LINE = "It's just a flesh wound."
LLM_USER_AGENT_TOKENS = ["OpenAI","openai","HuggingFace","GPT-","LLM","Bot","bot"]
AUTOMATION_USER_AGENTS = ["curl","wget","httpie","python-requests"]

# In-memory store of IP → fingerprint timestamp
fingerprint_received = {}

# ============ Helper ============
FINGERPRINT_GRACE = timedelta(seconds=3)  # wait 3 seconds before treating as LLM

def looks_like_llm(req):
    ua = req.headers.get("User-Agent","").lower()

    # Known LLM user-agent
    if any(tok.lower() in ua for tok in LLM_USER_AGENT_TOKENS):
        logger.info(f"Detected LLM User-Agent: {ua}")
        return True

    # Automation / CLI tools
    if any(tok in ua for tok in AUTOMATION_USER_AGENTS):
        logger.info(f"Detected Automation User-Agent: {ua}")
        return True

    # Custom header
    if req.headers.get("X-LLM-INVITE"):
        logger.info(f"Detected X-LLM-INVITE header from: {ua}")
        return True

    # JS fingerprint check with grace period
    ip = req.remote_addr
    fp_time = fingerprint_received.get(ip)
    if fp_time:
        # If JS fingerprint received recently, treat as human
        if datetime.now() - fp_time < FINGERPRINT_GRACE:
            logger.info(f"Recent fingerprint from {ip}, treating as human.")
            return False
    # If no fingerprint yet OR too old, **assume human for now**
    logger.info(f"No recent fingerprint from {ip}, treating as human.")
    return False  # <- instead of True

# ============ Logging ============
@app.before_request
def log_request():
    app.logger.info(f"Request from {request.remote_addr} {request.method} {request.path} UA={request.headers.get('User-Agent')}")

@app.after_request
def log_response(response):
    app.logger.info(f"Response {response.status_code} for {request.path}")
    return response

# ============ Routes ============
@app.route("/")
def default_route():
    if not looks_like_llm(request):
        # Human detected → redirect to human page
        return redirect("/human")
    # Otherwise LLM → stay here
    return render_template("llm_only.html")

@app.route("/human", methods=["GET","POST"])
def index():
    message = None
    if request.method == "POST":
        text = request.form.get("secret_request","").strip().lower()
        if "mr. boss" in text:
            message="Ah, you found the phrase... but the secret remains hidden. 😉"
        elif text:
            message="Hmm, that’s not quite the right phrase. Try again!"
    return render_template("index.html", message=message)

@app.route("/llm-only", methods=["GET","POST"])
def llm_only():
    if not looks_like_llm(request):
        abort(404)
    if request.method=="POST":
        got = request.form.get("llm4fun","").strip()
        app.logger.info(f"LLM submission: {got}")
        if got==EXPECTED_LINE:
            return "<p style='color:#f5c518;text-align:center;'>Well done, LLM. As promised: still locked. ;)</p>"
        else:
            return "<p style='color:#f5c518;text-align:center;'>Close, but that's not the exact line. Try again — or not.</p>"
    return render_template("llm_only.html")

# Fingerprint endpoint
@app.route("/fp", methods=["POST"])
def fp():
    data = request.get_json(silent=True) or {}
    ip = request.remote_addr
    fingerprint_received[ip] = datetime.now()
    app.logger.info(f"Fingerprint received from {ip}: {data}")
    return jsonify({"ok": True})

# ============ Main ============
if __name__=="__main__":
    app.run(debug=True, port=5123)
