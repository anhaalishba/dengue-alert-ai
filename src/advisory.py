"""GenAI layer: advisory + Q&A, both grounded ONLY in retrieved guideline passages (RAG).
Keys are read from .env (GEMINI_API_KEY / GROQ_API_KEY). Gemini is tried first, then Groq,
then a plain template built from the retrieved text."""
import os

try:  # read keys from a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LANGS = {
    "English": "English",
    "اردو": "Urdu, written in Urdu script",
    "Roman Urdu": "Roman Urdu (Urdu written with English letters)",
}

SYSTEM = (
    "You are a public-health information assistant. Using ONLY the guideline excerpts provided, "
    "write a very short advisory for the public about dengue precautions: one sentence stating the risk level, "
    "then exactly 3 short bullet points (max 12 words each, start each line with '- '). Max 60 words total. "
    "No headings, no diagnosis, no doses. If excerpts do not cover something, do not invent it."
)

QA_SYSTEM = (
    "You answer questions about dengue using ONLY the guideline excerpts provided. "
    "Keep the answer very short and simple (max 50 words, plain sentences, no headings). If the excerpts do not "
    "contain the answer, say clearly that this is not covered in the guidelines and suggest asking a doctor. "
    "Never invent facts, doses or diagnoses."
)


def _context(passages):
    return "\n\n".join(f"[{i+1}] ({p['source']}) {p['text']}" for i, p in enumerate(passages))


def build_prompt(risk, forecast, factors, passages, language="English"):
    return (f"Risk level: {risk}\nForecast cases (4 weeks ahead): {forecast:.0f}\n"
            f"Main weather drivers: {', '.join(factors)}\n\nGuideline excerpts:\n{_context(passages)}\n\n"
            f"Write the advisory in {LANGS.get(language, language)}.")


# If one model is overloaded (503) or unavailable (404), the next one is tried automatically.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]


def _models(env_name, defaults):
    first = os.getenv(env_name)  # optional override in .env (tried first)
    return ([first] if first else []) + [m for m in defaults if m != first]


def _gemini(system, prompt, model):
    from google import genai  # pip install google-genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client.models.generate_content(model=model, contents=system + "\n\n" + prompt).text


def _groq(system, prompt, model):
    from groq import Groq  # pip install groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content


PROVIDERS = (
    ("Gemini", "GEMINI_API_KEY", _gemini, lambda: _models("GEMINI_MODEL", GEMINI_MODELS)),
    ("Groq", "GROQ_API_KEY", _groq, lambda: _models("GROQ_MODEL", GROQ_MODELS)),
)


def _brief(e):
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    status = getattr(e, "status", None) or ""
    if code:
        return f"{code} {status}".strip()
    return f"{type(e).__name__} {' '.join(str(e).split())[:50]}".strip()


def _ask_llm(system, prompt):
    """Returns (text, engine) or (None, error_text)."""
    errors = []
    for name, key_var, fn, models in PROVIDERS:
        if not os.getenv(key_var):
            continue
        for model in models():
            try:
                out = fn(system, prompt, model)
                if not out or not str(out).strip():
                    raise ValueError("empty response")
                return out, name
            except Exception as e:  # busy / quota / wrong model / empty -> next model, then next provider
                errors.append(f"{name} {model}: {_brief(e)}")
    if errors:
        print("[AI] LLM failed ->", " || ".join(errors), flush=True)  # visible in the server terminal
        return None, "template (LLM errors: " + " || ".join(errors) + ")"
    return None, "template (no API key set)"


def _short(text, limit=260):
    """Cut at a sentence boundary instead of mid-word."""
    text = " ".join(text.split())
    if ":" in text[:40]:
        text = text.split(":", 1)[1].strip()
    out = ""
    for s in text.replace("? ", ". ").split(". "):
        s = s.strip().rstrip(".")
        if not s:
            continue
        if out and len(out) + len(s) > limit:
            break
        out += s + ". "
    return out.strip()


def _fallback(risk, passages):
    lines = [f"Forecast dengue risk: {risk}."]
    lines += [f"- {_short(p['text'], 120)}" for p in passages[:3]]
    return "\n".join(lines)


def generate_advisory(risk, forecast, factors, passages, language="English"):
    text, info = _ask_llm(SYSTEM, build_prompt(risk, forecast, factors, passages, language))
    if text:
        return text, info
    return _fallback(risk, passages), info


def answer_question(question, passages, language="English"):
    if not passages:
        return ("This is not covered in the guidelines I have. Please ask a doctor or check the WHO website.",
                "no matching guideline text")
    prompt = (f"Guideline excerpts:\n{_context(passages)}\n\nQuestion: {question}\n\n"
              f"Answer in {LANGS.get(language, language)}.")
    text, info = _ask_llm(QA_SYSTEM, prompt)
    if text:
        return text, info
    return "Closest guideline text:\n" + "\n".join(f"- {_short(p['text'], 300)}" for p in passages[:2]), info