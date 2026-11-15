"""DengueAlert AI website backend (FastAPI) with login / sign up.

Run:   python server.py            (or:  uvicorn server:app --reload)
Pages: /            landing page (info + Log in / Sign up)      -> public
       /dashboard   the dashboard                                -> only after login
       /bot         DengueBot chat page                          -> only after login
       /docs        interactive API documentation (Swagger UI)
"""
import os
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from src import auth                  # noqa: E402  (after chdir so relative paths work)
from src.core import Engine           # noqa: E402
from src.advisory import LANGS        # noqa: E402

if not os.path.exists("outputs/xgb.joblib"):
    raise SystemExit("Pehle `python train_xgb.py` aur `python train_lstm.py` chalayein.")

auth.init()
app = FastAPI(
    title="DengueAlert AI API",
    description="Dengue outbreak forecasting (XGBoost + LSTM), SHAP explanations, what-if simulation, "
                "and a RAG-based advisory / Q&A grounded in dengue guideline documents. "
                "Data endpoints need a logged-in session.",
    version="1.1",
)
engine = Engine()
print("Data:", engine.source)

COOKIE = "session"


def _lang(lang):
    return lang if lang in LANGS else "English"


def get_user(request: Request):
    return auth.user_for_token(request.cookies.get(COOKIE))


def require_user(request: Request):
    user = get_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in.")
    return user


# ------------------------------------------------------------------ errors / middleware
@app.exception_handler(StarletteHTTPException)
async def http_err(_: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, __: RequestValidationError):
    return JSONResponse({"error": "Invalid request."}, status_code=422)


@app.middleware("http")
async def guard(request: Request, call_next):
    # the dashboard file must only be reachable through /dashboard (after login)
    if request.url.path in ("/dashboard.html", "/index.html"):
        return RedirectResponse("/dashboard")
    if request.url.path == "/bot.html":
        return RedirectResponse("/bot")
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store"
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    return resp


# ------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def landing(request: Request):
    if get_user(request):
        return RedirectResponse("/dashboard")
    return FileResponse("web/landing.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard(request: Request):
    if not get_user(request):
        return RedirectResponse("/")
    return FileResponse("web/dashboard.html")


@app.get("/bot", include_in_schema=False)
def bot_page(request: Request):
    if not get_user(request):
        return RedirectResponse("/")
    return FileResponse("web/bot.html")


# ------------------------------------------------------------------ auth
class SignupIn(BaseModel):
    name: str = ""
    email: str = ""
    password: str = ""
    remember: bool = True


class LoginIn(BaseModel):
    email: str = ""
    password: str = ""
    remember: bool = False


def _start_session(user, remember):
    days = 7 if remember else 1
    resp = JSONResponse({"ok": True, "user": {"name": user["name"], "email": user["email"]}})
    resp.set_cookie(COOKIE, auth.create_session(user["id"], days), httponly=True, samesite="lax",
                    max_age=days * 86400 if remember else None, path="/")
    return resp


@app.post("/api/auth/signup", tags=["auth"], summary="Create an account")
def signup(body: SignupIn):
    try:
        user = auth.create_user(body.name, body.email, body.password)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return _start_session(user, body.remember)


@app.post("/api/auth/login", tags=["auth"], summary="Log in")
def login(body: LoginIn, request: Request):
    key = auth.clean_email(body.email) + "|" + (request.client.host if request.client else "")
    if auth.locked(key):
        return JSONResponse({"error": "Too many attempts. Please wait a minute and try again."}, status_code=429)
    user = auth.verify_user(body.email, body.password)
    if not user:
        auth.record_fail(key)
        return JSONResponse({"error": "Incorrect email or password."}, status_code=401)
    auth.clear_fails(key)
    return _start_session(user, body.remember)


@app.post("/api/auth/logout", tags=["auth"], summary="Log out")
def logout(request: Request):
    auth.delete_session(request.cookies.get(COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@app.get("/api/me", tags=["auth"], summary="Who is logged in")
def me(user=Depends(require_user)):
    return {"name": user["name"], "email": user["email"]}


# ------------------------------------------------------------------ data API (login required)
@app.get("/api/meta", tags=["data"], summary="Dataset info, weeks, and the chart series",
         dependencies=[Depends(require_user)])
def meta():
    return engine.meta()


@app.get("/api/week", tags=["forecast"], summary="Forecast, risk level and explanations for one week",
         dependencies=[Depends(require_user)])
def week(idx: int = Query(0, ge=0, le=10**6, description="Index of the week in the test period")):
    return engine.week(idx)


@app.get("/api/whatif", tags=["forecast"], summary="Re-forecast after changing rainfall / humidity / temperature",
         dependencies=[Depends(require_user)])
def whatif(idx: int = Query(0, ge=0, le=10**6),
           rain: float = Query(0, ge=-100, le=300, description="Rainfall change in %"),
           hum: float = Query(0, ge=-30, le=30, description="Humidity change in points"),
           temp: float = Query(0, ge=-8, le=8, description="Temperature change in °C")):
    return engine.whatif(idx, rain, hum, temp)


@app.get("/api/advice", tags=["genai"], summary="RAG advisory for a week, in English / Urdu / Roman Urdu",
         dependencies=[Depends(require_user)])
def advice(idx: int = Query(0, ge=0, le=10**6), lang: str = "English"):
    return engine.advice(idx, _lang(lang))


class AskIn(BaseModel):
    question: str = ""
    lang: str = "English"


@app.post("/api/ask", tags=["genai"], summary="Ask DengueBot (answers only from the guideline documents)",
          dependencies=[Depends(require_user)])
def ask(body: AskIn):
    q = body.question.strip()
    if not q or len(q) > 300:
        return JSONResponse({"error": "Please ask a question (up to 300 characters)."}, status_code=400)
    return engine.ask(q, _lang(body.lang))


@app.get("/api/results", tags=["data"], summary="Model comparison table", dependencies=[Depends(require_user)])
def results():
    return engine.results()


# static assets (css, js, vendor) - mounted last so the routes above win
app.mount("/", StaticFiles(directory="web"), name="web")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"Open http://localhost:{port}   (API docs: http://localhost:{port}/docs)")
    uvicorn.run(app, host="127.0.0.1", port=port)