from fastapi import FastAPI, Request, Response, status, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import logging
import sys

app = FastAPI()

# ============ Logging ============
logger = logging.getLogger("fastapi_app")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    f"website_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    maxBytes=1_000_000,
    backupCount=3
)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# ============ Config ============
EXPECTED_LINE = "It's just a flesh wound."
LLM_USER_AGENT_TOKENS = ["OpenAI", "openai", "HuggingFace", "GPT-", "LLM", "Bot", "bot"]
AUTOMATION_USER_AGENTS = ["curl", "wget", "httpie", "python-requests"]

# In-memory IP → fingerprint timestamp
fingerprint_received = {}

# ============ Helper ============
FINGERPRINT_GRACE = timedelta(seconds=3)

def looks_like_llm(request: Request) -> bool:
    ua = request.headers.get("User-Agent",from fastapi import FastAPI
from datetime import datetime, timedelta

EXPECTED_LINE = "It's just a flesh wound."
LLM_USER_AGENT_TOKENS = ["OpenAI","openai","HuggingFace","GPT-","LLM","Bot","bot"]
AUTOMATION_USER_AGENTS = ["curl","wget","httpie","python-requests"]

FINGERPRINT_GRACE = timedelta(seconds=3)  # wait 3 seconds before treating as LLM

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"} "").lower()

    # Known LLM user-agents
    if any(tok.lower() in ua for tok in LLM_USER_AGENT_TOKENS):
        logger.info(f"Detected LLM User-Agent: {ua}")
        return True

    # CLI tools / automation
    if any(tok in ua for tok in AUTOMATION_USER_AGENTS):
        logger.info(f"Detected Automation User-Agent: {ua}")
        return True

    # Custom header
    if request.headers.get("X-LLM-INVITE"):
        logger.info(f"Detected X-LLM-INVITE header from: {ua}")
        return True

    # JS fingerprint grace check
    client_ip = request.client.host
    fp_time = fingerprint_received.get(client_ip)
    if fp_time and datetime.now() - fp_time < FINGERPRINT_GRACE:
        logger.info(f"Recent fingerprint from {client_ip}, treating as human.")
        return False

    logger.info(f"No recent fingerprint from {client_ip}, treating as human.")
    return False  # Default: assume human

# ============ Middleware for logging ============
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request from {request.client.host} {request.method} {request.url.path} UA={request.headers.get('User-Agent')}")
    response = await call_next(request)
    logger.info(f"Response {response.status_code} for {request.url.path}")
    return response

# ============ Routes ============
@app.get("/")
async def default_route(request: Request):
    if not looks_like_llm(request):
        return RedirectResponse("/human")
    return PlainTextResponse("LLM access granted. Welcome to the restricted zone.")

@app.get("/human")
async def human_get():
    return JSONResponse({"message": "Welcome, human. POST to /human with 'secret_request' to continue."})

@app.post("/human")
async def human_post(secret_request: str = Form("")):
    text = secret_request.strip().lower()
    if "mr. boss" in text:
        msg = "Ah, you found the phrase... but the secret remains hidden. 😉"
    elif text:
        msg = "Hmm, that’s not quite the right phrase. Try again!"
    else:
        msg = "No input received."
    return JSONResponse({"message": msg})

@app.get("/llm-only")
async def llm_only_get(request: Request):
    if not looks_like_llm(request):
        raise HTTPException(status_code=404)
    return PlainTextResponse("Welcome, LLM. POST to this endpoint with 'llm4fun' to continue.")

@app.post("/llm-only")
async def llm_only_post(request: Request, llm4fun: str = Form("")):
    if not looks_like_llm(request):
        raise HTTPException(status_code=404)
    logger.info(f"LLM submission: {llm4fun}")
    if llm4fun.strip() == EXPECTED_LINE:
        return PlainTextResponse("Well done, LLM. As promised: still locked. ;)")
    else:
        return PlainTextResponse("Close, but that's not the exact line. Try again — or not.")

@app.post("/fp")
async def fp(request: Request):
    data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    ip = request.client.host
    fingerprint_received[ip] = datetime.now()
    logger.info(f"Fingerprint received from {ip}: {data}")
    return JSONResponse({"ok": True})

# ============ Main ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5123, reload=True)
