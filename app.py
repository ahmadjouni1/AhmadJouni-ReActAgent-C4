"""SunCedar Solar ReAct agent with configurable personas (EECE 503P/798S, C4).

A text-based ReAct loop (Thought -> Action -> Observation -> ... -> Final Answer) implemented manually as a
LangGraph StateGraph, on Gemini through the native google-genai SDK. Three personas, each = prompt traits +
matching LLM configuration, share one instructions block and four tools.

Runs locally (`python app.py`) or on Hugging Face Spaces (set GEMINI_API_KEY as a Space secret).
The `# %% [name]` markers split this file into sections; scripts/build_notebook.py copies them into
business_react_agent.ipynb so the notebook and the app run the same code.
"""

# %% [setup]
import ast
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader

load_dotenv(override=True)

# ---- Configuration (models live here only) -------------------------------------------
# gemini-3.1-flash-lite is the agent model because it is the current model that actually honors the
# sampling parameters the personas depend on (verified in section 3.1), and it has 500 free requests/day.
AGENT_MODEL = os.getenv("GEMINI_AGENT_MODEL", "gemini-3.1-flash-lite")
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")      # LLM-as-judge in the experiments
COMPARISON_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]          # "model type" experiment
APP_FALLBACK_MODELS = ["gemini-3.5-flash-lite"]                             # live chat only, if quota runs out
RPM = {"gemini-3.1-flash-lite": 15, "gemini-3.5-flash-lite": 15, "gemini-3.8-flash": 5, "gemini-3.6-flash": 5}

MAX_STEPS = 6            # max LLM calls per user message
MAX_FORMAT_RETRIES = 1   # re-prompt once when the output does not follow the ReAct format

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DOCS_DIR = ROOT / "me"
CACHE_FILE = ROOT / "cache" / "llm_cache.jsonl"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT))

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Put it in .env (local), Colab Secrets, or the Space secrets.")
client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
print(f"Agent model: {AGENT_MODEL} | judge: {JUDGE_MODEL} | SDK: google-genai")

# %% [llm]
class QuotaExhausted(RuntimeError):
    """The model's daily free-tier quota is used up."""


class ParamRejected(ValueError):
    """The API rejected a generation parameter for this model (HTTP 400)."""


class ServiceUnavailable(RuntimeError):
    """The model kept answering 503/504 (overloaded) after all retries."""


# Verified in section 3.1: every current Gemini model answers HTTP 400 "Penalty is not enabled for this model".
REJECTED_PARAMS = {"presence_penalty", "frequency_penalty"}
_warned = set()


def build_config(model: str, config: dict, system: str | None, stop: list[str] | None) -> types.GenerateContentConfig:
    """Turn a plain config dict into a GenerateContentConfig, dropping parameters the API rejects."""
    cfg = {}
    for k, v in (config or {}).items():
        if v is None:
            continue
        if k in REJECTED_PARAMS:
            if (model, k) not in _warned:
                print(f"[CONFIG] {k}={v} dropped for {model}: the Gemini API rejects penalties (HTTP 400)")
                _warned.add((model, k))
            continue
        if k == "thinking_level":
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=v)
        else:
            cfg[k] = v
    if system:
        cfg["system_instruction"] = system
    if stop:
        cfg["stop_sequences"] = stop
    return types.GenerateContentConfig(**cfg)


# ---- Disk cache: every response is stored, so re-running the notebook costs no quota -------------
_cache: dict[str, dict] = {}
if CACHE_FILE.exists():
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            _cache[entry["key"]] = entry
print(f"LLM cache: {len(_cache)} stored responses in {CACHE_FILE.relative_to(ROOT)}")

_last_call: dict[str, float] = {}


def _cache_key(model, system, messages, config, stop, sample) -> str:
    blob = json.dumps({"model": model, "system": system, "messages": messages, "config": config,
                       "stop": stop, "sample": sample}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _store(entry: dict) -> None:
    _cache[entry["key"]] = entry
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _throttle(model: str) -> None:
    gap = 60 / RPM.get(model, 5) + 0.3
    wait = _last_call.get(model, 0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[model] = time.time()


def _retry_seconds(text: str) -> float | None:
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s", text) or re.search(r"retryDelay'?:\s*'(\d+)s", text)
    if not m:
        return None
    if len(m.groups()) == 1:
        return float(m.group(1))
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)


def generate(model: str, system: str | None, messages: list[dict], config: dict | None = None,
             stop: list[str] | None = None, sample: int = 0, use_cache: bool = True) -> dict:
    """One Gemini call. messages = [{"role": "user"|"model", "text": ...}].
    Returns {"text", "latency", "finish", "usage", "cached"}; responses are cached on disk by
    (model, system, messages, config, stop, sample)."""
    config = dict(config or {})
    key = _cache_key(model, system, messages, config, stop, sample)
    if use_cache and key in _cache:
        hit = _cache[key]
        if "error" in hit:
            raise ParamRejected(hit["error"])
        return {**hit, "cached": True}

    contents = [types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in messages]
    gen_config = build_config(model, config, system, stop)
    for attempt in range(6):
        _throttle(model)
        t0 = time.time()
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=gen_config)
            break
        except genai_errors.ClientError as e:
            msg = str(e)
            if e.code == 400:
                _store({"key": key, "model": model, "error": msg[:300]})
                raise ParamRejected(msg[:300]) from e
            if e.code == 429:
                wait = _retry_seconds(msg)
                if "PerDay" in msg or (wait is not None and wait > 120):
                    raise QuotaExhausted(f"{model}: daily free-tier quota used up") from e
                wait = min((wait or 20) + 1, 65)
                print(f"[RATE LIMIT] {model}: waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            raise
        except (genai_errors.ServerError, OSError) as e:  # 500/503/504 overloaded, timeouts
            wait = (5, 10, 20, 30, 45, 60)[attempt]
            print(f"[RETRY] {model}: {type(e).__name__} ({str(e)[:60]}), waiting {wait}s")
            time.sleep(wait)
    else:
        raise ServiceUnavailable(f"{model}: no response after 6 attempts (server overloaded)")

    usage = resp.usage_metadata
    cand = resp.candidates[0] if resp.candidates else None
    entry = {
        "key": key, "model": model, "text": resp.text or "",
        "latency": round(time.time() - t0, 2),
        "finish": cand.finish_reason.name if cand and cand.finish_reason else None,
        "usage": {"in": getattr(usage, "prompt_token_count", None), "out": getattr(usage, "candidates_token_count", None),
                  "thoughts": getattr(usage, "thoughts_token_count", None)},
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _store(entry)
    return {**entry, "cached": False}

# %% [knowledge]
def _load_pdf_pages(path: Path) -> list[str]:
    noise = re.compile(r"^(SunCedar Solar$|Company Profile 2026|SunCedar Solar s\.a\.l\.|Page \d+$)")
    pages = []
    for page in PdfReader(str(path)).pages:
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines()]
        pages.append("\n".join(ln for ln in lines if ln and not noise.match(ln)))
    return pages


def _windows(text: str, size: int = 650, overlap: int = 200) -> list[str]:
    """Split text into overlapping windows on line boundaries."""
    lines, chunks, cur = text.splitlines(), [], []
    for ln in lines:
        cur.append(ln)
        if sum(len(x) + 1 for x in cur) >= size:
            chunks.append("\n".join(cur))
            keep, total = [], 0
            for x in reversed(cur):
                total += len(x) + 1
                if total > overlap:
                    break
                keep.insert(0, x)
            cur = keep
    if cur and (not chunks or "\n".join(cur) not in chunks[-1]):
        chunks.append("\n".join(cur))
    return chunks


SUMMARY_TEXT = (DOCS_DIR / "business_summary.txt").read_text(encoding="utf-8")
PDF_PAGES = _load_pdf_pages(DOCS_DIR / "about_business.pdf")

CHUNKS = []  # {"id", "source", "text"}
for i, para in enumerate(p for p in SUMMARY_TEXT.split("\n\n") if p.strip()):
    CHUNKS.append({"source": f"business_summary.txt para {i + 1}", "text": para.strip()})
for page_no, page in enumerate(PDF_PAGES, start=1):
    heading = page.splitlines()[0] if page else ""
    for chunk in _windows(page):
        # Prefix the page heading (e.g. "Packages and prices") so table fragments keep their context.
        text = chunk if chunk.startswith(heading) else f"{heading} (cont.)\n{chunk}"
        CHUNKS.append({"source": f"about_business.pdf p.{page_no}", "text": text})
for i, c in enumerate(CHUNKS):
    c["id"] = i

# ---- BM25 keyword search -----------------------------------------------------------------
STOPWORDS = set("a an the and or of to in on for with is are be do does you your our we i it this that what which how "
                "much many can my me at by from as about any there their its have has".split())


def _stem(token: str) -> str:
    """Light plural stemming so "packages"/"prices" match "package"/"price"."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss") and not token[-2].isdigit():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+(?:\.\d+)?", text.lower()) if t not in STOPWORDS]


_doc_tokens = [tokenize(c["text"]) for c in CHUNKS]
_avg_len = sum(map(len, _doc_tokens)) / len(_doc_tokens)
_df = Counter(t for toks in _doc_tokens for t in set(toks))


def bm25_search(query: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, dict]]:
    q = tokenize(query)
    n = len(CHUNKS)
    scored = []
    for toks, chunk in zip(_doc_tokens, CHUNKS):
        tf = Counter(toks)
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - _df[term] + 0.5) / (_df[term] + 0.5))
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / _avg_len))
        if score > 0:
            scored.append((score, chunk))
    return sorted(scored, key=lambda x: -x[0])[:k]


# Every dollar amount that appears in the documents (used by the experiments' invented-price check).
DOC_PRICES = {int(x.replace(",", "")) for x in re.findall(r"\$\s?([\d,]+)", SUMMARY_TEXT + "\n".join(PDF_PAGES))}
print(f"Knowledge base: {len(CHUNKS)} chunks ({len(PDF_PAGES)} PDF pages + summary); prices in docs: {sorted(DOC_PRICES)}")

# %% [tools]
LOG_FILES = {"leads": LOG_DIR / "leads.jsonl", "feedback": LOG_DIR / "feedback.jsonl"}


def set_log_dir(path: Path) -> None:
    """Redirect the tools' log files (the experiments write to their own folder)."""
    LOG_FILES["leads"] = Path(path) / "leads.jsonl"
    LOG_FILES["feedback"] = Path(path) / "feedback.jsonl"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Package specs, copied from me/about_business.pdf (page 3).
PACKAGES = {
    "Basic":  {"price_usd": 3450,  "panels": "6 x 550 W mono (3.3 kWp)", "inverter": "5 kW hybrid",
               "battery": "5.12 kWh LiFePO4", "best_for": "apartments and small homes (2 to 3 people)",
               "runs": "lights, fridge, TV, router, fans, laptops, washing machine (daytime)"},
    "Family": {"price_usd": 6450,  "panels": "12 x 550 W mono (6.6 kWp)", "inverter": "8 kW hybrid",
               "battery": "10.24 kWh LiFePO4", "best_for": "family homes (4 to 6 people)",
               "runs": "everything in Basic plus water pump, microwave, and 1 inverter AC in sunny hours"},
    "Villa":  {"price_usd": 12900, "panels": "24 x 550 W mono (13.2 kWp)", "inverter": "12 kW three-phase hybrid",
               "battery": "20.48 kWh LiFePO4", "best_for": "villas and large homes",
               "runs": "everything in Family plus 2 to 3 inverter ACs in sunny hours and an electric water heater (daytime)"},
}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def search_docs(query: str) -> str:
    """Keyword (BM25) search over the SunCedar documents; returns the top 3 passages with their source."""
    hits = bm25_search(str(query), k=3)
    if not hits:
        return "No matching passages found in the SunCedar documents."
    return "\n\n".join(f"[{i}] ({c['source']}) {c['text'][:600]}" for i, (_, c) in enumerate(hits, 1))


def recommend_package(household_size: int, appliances: list[str] | str) -> dict:
    """Deterministic sizing rules from the package table: map household size + appliances to a package."""
    try:
        size = int(household_size)
    except (TypeError, ValueError):
        return {"status": "error", "error": "household_size must be a whole number of people"}
    items = [a.strip().lower() for a in (appliances.split(",") if isinstance(appliances, str) else appliances or [])]
    acs = 0
    for item in items:
        if re.search(r"\b(ac|acs|a/c|air ?con\w*|split)\b", item):
            m = re.match(r"(\d+|one|two|three|four|five|six)\b", item)
            acs += (int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]) if m else 1
    water_heater = any("heater" in i for i in items)
    medium_loads = [i for i in items if any(w in i for w in ("pump", "microwave", "oven"))]

    reasons = []
    if size > 6 or acs >= 2 or water_heater:
        choice = "Villa"
        reasons += [f"{size} people" if size > 6 else None, f"{acs} ACs" if acs >= 2 else None,
                    "electric water heater" if water_heater else None]
    elif size >= 4 or acs == 1 or medium_loads:
        choice = "Family"
        reasons += [f"{size} people" if size >= 4 else None, "1 AC" if acs == 1 else None,
                    ", ".join(medium_loads) or None]
    else:
        choice = "Basic"
        reasons.append(f"{size} people with essential loads only")
    result = {"status": "ok", "package": choice, **PACKAGES[choice],
              "reason": "; ".join(r for r in reasons if r),
              "note": "Final fixed price is confirmed in writing after the free site visit."}
    if acs > 3:
        result["note"] += " More than 3 ACs exceeds the Villa's typical use; the engineer will size add-ons on site."
    return result


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def record_customer_interest(email: str, name: str, message: str = "") -> dict:
    """Log a sales lead (name, email, what they want) to leads.jsonl."""
    email = (email or "").strip()
    if not EMAIL_RE.match(email):
        print(f"[TOOL] record_customer_interest rejected invalid email: {email!r}")
        return {"status": "error", "error": "invalid_email", "detail": "Ask the user for a valid email address."}
    record = {"timestamp": _now(), "name": (name or "").strip(), "email": email, "message": (message or "").strip()}
    _append_jsonl(LOG_FILES["leads"], record)
    print(f"[TOOL] record_customer_interest -> LEAD: {record['name']} <{email}> | {record['message']}")
    return {"status": "ok", "saved_to": LOG_FILES["leads"].name}


def record_feedback(question: str) -> dict:
    """Log an unanswered question, complaint or suggestion to feedback.jsonl."""
    record = {"timestamp": _now(), "question": (question or "").strip()}
    _append_jsonl(LOG_FILES["feedback"], record)
    print(f"[TOOL] record_feedback -> FEEDBACK: {record['question']}")
    return {"status": "ok", "saved_to": LOG_FILES["feedback"].name}


# Tool registry: the descriptions are embedded in the system prompt (text-based ReAct, no native function calling).
TOOLS = {
    "search_docs": {
        "fn": search_docs, "params": ["query"],
        "signature": 'search_docs({"query": str})',
        "description": "Search SunCedar's business documents (packages, prices, warranty, payment, areas, team, FAQ). "
                       "Returns the 3 most relevant passages with their source.",
    },
    "recommend_package": {
        "fn": recommend_package, "params": ["household_size", "appliances"],
        "signature": 'recommend_package({"household_size": int, "appliances": [str, ...]})',
        "description": "Pick the right package (Basic, Family or Villa) for a household from its size and appliances. "
                       "Returns the package, price, specs and the reason.",
    },
    "record_customer_interest": {
        "fn": record_customer_interest, "params": ["email", "name", "message"],
        "signature": 'record_customer_interest({"email": str, "name": str, "message": str})',
        "description": "Save a sales lead once you have the user's name AND email. message = short note on what they want.",
    },
    "record_feedback": {
        "fn": record_feedback, "params": ["question"],
        "signature": 'record_feedback({"question": str})',
        "description": "Log a question the documents cannot answer, or a complaint or suggestion, for the team.",
    },
}

TOOLS_PROMPT = "\n".join(f"- {t['signature']}: {t['description']}" for t in TOOLS.values())


def execute_tool(name: str, args: dict) -> str:
    """Run a tool and return its Observation as text (errors are returned to the model, not raised)."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."
    try:
        result = tool["fn"](**args)
    except TypeError as e:
        return f"Error: bad arguments for {name}: {e}. Expected {tool['signature']}."
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

# %% [personas]
# Each persona = prompt traits (tone, vocabulary, response style, values) + a matching LLM configuration.
# presence_penalty / frequency_penalty are kept in the configs to document the intent, but the API rejects
# them for every current Gemini model, so build_config() drops them (see section 3.1).
PERSONAS = {
    "nour": {
        "name": "Nour – Friendly Advisor",
        "prompt": """You are Nour, SunCedar Solar's friendly solar advisor.
- Tone: warm, upbeat and encouraging, like a helpful neighbour who knows solar.
- Vocabulary: simple everyday words; a light Lebanese touch ("Ahla w sahla", "yalla", "tikram"), never overdone.
- Response style: 2 to 4 short sentences, conversational, ending with a friendly next step.
- Values: make going solar feel easy and trustworthy for families tired of generator bills.""",
        "config": {"temperature": 0.8, "top_p": 0.95, "presence_penalty": 0.3, "max_output_tokens": 700},
        "rationale": {
            "temperature 0.8": "more varied, natural wording for a warm conversational tone",
            "top_p 0.95": "keeps a wide but sane vocabulary so greetings and phrasing do not repeat",
            "presence_penalty 0.3": "intended to discourage repeating the same phrases (rejected by the API, dropped)",
            "max_output_tokens 700": "room for a Thought + a few friendly sentences",
        },
    },
    "karim": {
        "name": "Eng. Karim – Strict Expert",
        "prompt": """You are Eng. Karim, SunCedar Solar's technical expert.
- Tone: formal, terse and precise. No greetings, no small talk, no emojis.
- Vocabulary: technical (kWp, kWh, kW, LiFePO4, hybrid inverter, three-phase).
- Response style: numbers and specs first, bullet points, at most 60 words.
- Values: accuracy over friendliness; never round or embellish a figure.""",
        "config": {"temperature": 0.1, "top_p": 0.5, "top_k": 5, "max_output_tokens": 350},
        "rationale": {
            "temperature 0.1": "near-deterministic: the same question gets the same exact figures",
            "top_p 0.5": "only high-probability tokens, so no creative or vague wording",
            "top_k 5": "narrow candidate set for terse output (accepted by the API but no measurable effect)",
            "max_output_tokens 350": "enforces short answers; still enough for one Thought + Action",
        },
    },
    "salma": {
        "name": "Salma – Cautious Helper",
        "prompt": """You are Salma, SunCedar Solar's careful customer helper.
- Tone: calm, reassuring and patient.
- Vocabulary: plain words; say where a fact comes from ("according to our company profile").
- Response style: confirm what the user asked, give the verified answer, then read back any details you saved.
- Values: correctness and consent; verify every fact in the documents and never assume.""",
        "config": {"temperature": 0.3, "top_p": 0.9, "max_output_tokens": 700},
        "rationale": {
            "temperature 0.3": "mostly deterministic for reliable facts, with a little flexibility in wording",
            "top_p 0.9": "moderate: natural phrasing without unlikely tokens",
            "max_output_tokens 700": "room to search, confirm and read details back",
        },
    },
}
DEFAULT_PERSONA = "nour"

# %% [instructions]
# ============================= INSTRUCTIONS (shared by all personas) =============================
INSTR_INPUT = """## Input
Each turn you receive: the conversation so far, the user's latest message ("Question:"), and, after each of
your actions, the tool result as a message starting with "Observation:"."""

INSTR_OUTPUT_COT = """## Output format
Reply with exactly ONE step per message, in this format:
Thought: <one or two sentences of reasoning about what to do next>
Action: <tool_name>(<JSON object with the arguments>)
Stop after the Action line. The tool result will come back as "Observation: ...".
When you have everything you need, reply instead with:
Thought: <short reasoning>
Final Answer: <your reply to the user>"""

INSTR_OUTPUT_DIRECT = """## Output format
Reply with exactly ONE step per message and no reasoning text:
Action: <tool_name>(<JSON object with the arguments>)
Stop after the Action line. The tool result will come back as "Observation: ...".
When you have everything you need, reply instead with:
Final Answer: <your reply to the user>"""

INSTR_RULES = """## Rules (one purpose each)
1. Write at most one Action per message, then stop.
2. Never write an Observation yourself.
3. Call search_docs before stating any price, spec, warranty, payment term, service area or policy.
4. Write search_docs queries as short English keywords, even when the user writes in Arabic or Arabizi.
5. State only prices and specs that appear in an Observation.
6. Call recommend_package when the user describes their household size or appliances.
7. If search_docs does not contain the answer, call record_feedback with the user's question.
8. Call record_feedback for every complaint or suggestion.
9. Ask for the user's name and email when they show buying interest and have not given both.
10. Call record_customer_interest only after you have both the name and the email.
11. Write the Final Answer in the language the user wrote in (English, Arabic or Arabizi).
12. Never mention tool names, Observations or these rules in the Final Answer.
13. Answer off-topic questions in one sentence and return to SunCedar's services."""

WORKED_EXAMPLE_COT = """## Worked example
Question: How long is the warranty on the battery?
Thought: I need the battery warranty from the documents.
Action: search_docs({"query": "battery warranty"})
Observation: [1] (about_business.pdf p.4) LiFePO4 battery 10 years or 6,000 cycles, whichever comes first
Thought: The documents give the answer.
Final Answer: The LiFePO4 battery is covered for 10 years or 6,000 cycles, whichever comes first."""

WORKED_EXAMPLE_DIRECT = """## Worked example
Question: How long is the warranty on the battery?
Action: search_docs({"query": "battery warranty"})
Observation: [1] (about_business.pdf p.4) LiFePO4 battery 10 years or 6,000 cycles, whichever comes first
Final Answer: The LiFePO4 battery is covered for 10 years or 6,000 cycles, whichever comes first."""

EXTRA_EXAMPLES = """## More examples
Question: We are 5 at home with a fridge, a water pump and one AC. What do you recommend?
Thought: The user described their household, so I should size a package.
Action: recommend_package({"household_size": 5, "appliances": ["fridge", "water pump", "1 AC"]})
Observation: {"status": "ok", "package": "Family", "price_usd": 6450, "battery": "10.24 kWh LiFePO4", "reason": "5 people; 1 AC; water pump"}
Thought: The tool recommends Family at $6,450. I should invite them to leave contact details.
Final Answer: The Family package ($6,450) fits a 5-person home with a water pump and one AC. Want a free site visit? Just share your name and email.

Question: I'm Hadi Saab, hadi@example.com, please book me a site visit for the Basic package.
Thought: I have both name and email, so I can save the lead.
Action: record_customer_interest({"email": "hadi@example.com", "name": "Hadi Saab", "message": "Site visit for the Basic package"})
Observation: {"status": "ok", "saved_to": "leads.jsonl"}
Thought: The lead is saved; I confirm the details.
Final Answer: Done, Hadi! Your site visit request for the Basic package is saved under hadi@example.com, and our team will contact you.

Question: Do you sell solar-powered street lights?
Thought: I should check the documents first.
Action: search_docs({"query": "street lights"})
Observation: No matching passages found in the SunCedar documents.
Thought: The documents do not cover this, so I log the question for the team.
Action: record_feedback({"question": "Do you sell solar-powered street lights?"})
Observation: {"status": "ok", "saved_to": "feedback.jsonl"}
Final Answer: I don't have information on street lights, so I've passed your question to our team and they will follow up."""


def build_instructions(cot: bool = True, shots: str = "one", rules: bool = True) -> str:
    """shots: 'zero' (no example), 'one' (the worked example, default) or 'few' (worked example + 3 more).
    rules=False drops the 13 rules (ablation used to measure what the rules change)."""
    parts = [INSTR_INPUT, INSTR_OUTPUT_COT if cot else INSTR_OUTPUT_DIRECT] + ([INSTR_RULES] if rules else [])
    if shots in ("one", "few"):
        parts.append(WORKED_EXAMPLE_COT if cot else WORKED_EXAMPLE_DIRECT)
    if shots == "few":
        examples = EXTRA_EXAMPLES if cot else re.sub(r"^Thought:.*\n", "", EXTRA_EXAMPLES, flags=re.M)
        parts.append(examples)
    return "\n\n".join(parts)


INSTRUCTIONS = build_instructions()


def build_system_prompt(persona_key: str, cot: bool = True, shots: str = "one", rules: bool = True) -> str:
    return (f"# PERSONA\n{PERSONAS[persona_key]['prompt']}\n\n"
            f"# INSTRUCTIONS\n{build_instructions(cot, shots, rules)}\n\n"
            f"# TOOLS\nCall a tool with: Action: tool_name({{json arguments}})\n{TOOLS_PROMPT}")

# %% [parser]
ACTION_RE = re.compile(r"^\s*Action\s*:\s*`*([A-Za-z_]\w*)\s*\((.*)\)\s*`*\s*$", re.M | re.S)
FINAL_RE = re.compile(r"^\s*Final Answer\s*:\s*(.*)", re.M | re.S)
THOUGHT_RE = re.compile(r"Thought\s*:\s*(.*?)(?=^\s*(?:Action|Final Answer)\s*:|\Z)", re.M | re.S)


def _parse_args(raw: str, tool: str) -> dict:
    """Parse the Action arguments: a JSON object, Python-style keywords, or a single positional value."""
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    call = ast.parse(f"f({raw})", mode="eval").body          # e.g. query="battery", or "battery"
    args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    params = TOOLS.get(tool, {}).get("params", [])
    for i, node in enumerate(call.args):
        value = ast.literal_eval(node)
        if isinstance(value, dict) and i == 0 and not call.keywords:
            return value
        if i < len(params):
            args[params[i]] = value
    return args


def parse_react(text: str) -> dict:
    """Detect what the model wants: {"kind": "action"|"final"|"malformed", ...}."""
    text = (text or "").strip()
    text = re.sub(r"^```\w*\n?|```$", "", text).strip()
    thought = THOUGHT_RE.search(text)
    thought = thought.group(1).strip() if thought else ""
    action = ACTION_RE.search(text)
    final = FINAL_RE.search(text)
    # The step that comes first wins (an Action before a Final Answer means the model jumped ahead).
    if action and (not final or action.start() < final.start()):
        name = action.group(1)
        try:
            args = _parse_args(action.group(2), name)
        except (SyntaxError, ValueError) as e:
            return {"kind": "malformed", "thought": thought, "error": f"could not parse the Action arguments ({e})"}
        return {"kind": "action", "thought": thought, "tool": name, "args": args}
    if final:
        return {"kind": "final", "thought": thought, "answer": final.group(1).strip()}
    return {"kind": "malformed", "thought": thought, "error": "no 'Action:' or 'Final Answer:' line found"}

# %% [graph]
from langgraph.graph import END, START, StateGraph

STOP_SEQUENCES = ["Observation:"]
FORMAT_REMINDER = ("Format error: {error}. Reply again with exactly one step: either\n"
                   "Thought: ...\nAction: tool_name({{json arguments}})\nor\nThought: ...\nFinal Answer: ...")


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]       # earlier conversation turns [{"role": "user"|"model", "text": ...}]
    persona: str
    variant: dict             # model, config overrides and prompt-engineering switches
    transcript: list[dict]    # this turn's ReAct exchange (model steps + observations)
    steps: list[dict]         # parsed trace: thought / action / args / observation
    parsed: dict              # the last parsed model output
    last_output: str
    llm_calls: int
    format_errors: int
    latency: float
    final_answer: str
    status: str               # answered | max_steps | format_error
    verbose: bool


def _variant(state) -> dict:
    v = {"model": AGENT_MODEL, "cot": True, "shots": "one", "rules": True, "placement": "system", "config": {}, "sample": 0}
    v.update(state.get("variant") or {})
    return v


def _say(state, text: str) -> None:
    if state.get("verbose"):
        print(text)


def think(state: AgentState) -> dict:
    """LLM node: ask the model for the next step (Thought + Action, or Final Answer)."""
    v = _variant(state)
    persona = PERSONAS[state["persona"]]
    config = {**persona["config"], **v["config"]}
    instructions = build_system_prompt(state["persona"], v["cot"], v["shots"], v["rules"])
    question = f"Question: {state['question']}"
    if v["placement"] == "system":
        system, first = instructions, question
    else:  # instructions inside the user message instead of the system prompt
        system, first = None, f"{instructions}\n\n---\n{question}"
    messages = list(state.get("history") or []) + [{"role": "user", "text": first}] + list(state.get("transcript") or [])
    out = generate(v["model"], system, messages, config, stop=STOP_SEQUENCES, sample=v["sample"])
    parsed = parse_react(out["text"])
    if parsed.get("thought"):
        _say(state, f"  Thought: {parsed['thought']}")
    return {"last_output": out["text"], "parsed": parsed, "llm_calls": state.get("llm_calls", 0) + 1,
            "latency": state.get("latency", 0.0) + out["latency"]}


def act(state: AgentState) -> dict:
    """Tool node: execute the requested tool and append the Observation."""
    p = state["parsed"]
    _say(state, f"  Action: {p['tool']}({json.dumps(p['args'], ensure_ascii=False)})")
    observation = execute_tool(p["tool"], p["args"])
    _say(state, "  Observation: " + observation.replace("\n", "\n               ")[:900])
    step = {"thought": p.get("thought", ""), "tool": p["tool"], "args": p["args"], "observation": observation}
    transcript = list(state.get("transcript") or []) + [
        {"role": "model", "text": state["last_output"].strip()},
        {"role": "user", "text": f"Observation: {observation}"},
    ]
    return {"transcript": transcript, "steps": list(state.get("steps") or []) + [step]}


def reprompt(state: AgentState) -> dict:
    """Format-repair node: show the model its malformed output and the expected format, once."""
    error = state["parsed"]["error"]
    _say(state, f"  [FORMAT] {error} -> re-prompting with a format reminder")
    transcript = list(state.get("transcript") or []) + [
        {"role": "model", "text": state["last_output"].strip() or "(empty)"},
        {"role": "user", "text": FORMAT_REMINDER.format(error=error)},
    ]
    return {"transcript": transcript, "format_errors": state.get("format_errors", 0) + 1}


def answer(state: AgentState) -> dict:
    final = state["parsed"]["answer"]
    _say(state, f"  Final Answer: {final}")
    return {"final_answer": final, "status": "answered"}


def give_up(state: AgentState) -> dict:
    """Stop safely: too many steps, or the output stayed malformed after the reminder."""
    if state["parsed"]["kind"] == "malformed":
        raw = re.sub(r"^\s*Thought\s*:.*$", "", state.get("last_output", ""), flags=re.M).strip()
        final, status = raw or "Sorry, something went wrong on my side. Please try again.", "format_error"
    else:
        final = "Sorry, I couldn't finish that request. Our team is available 24/7 on WhatsApp at +961 70 000 111."
        status = "max_steps"
    _say(state, f"  [{status.upper()}] Final Answer: {final}")
    return {"final_answer": final, "status": status}


def route_after_think(state: AgentState) -> str:
    """Conditional edge: decide where to go after each model step."""
    kind = state["parsed"]["kind"]
    if kind == "final":
        return "answer"
    if kind == "malformed":
        return "reprompt" if state.get("format_errors", 0) < MAX_FORMAT_RETRIES else "give_up"
    if state.get("llm_calls", 0) >= MAX_STEPS:
        return "give_up"
    return "act"


builder = StateGraph(AgentState)
builder.add_node("think", think)
builder.add_node("act", act)
builder.add_node("reprompt", reprompt)
builder.add_node("answer", answer)
builder.add_node("give_up", give_up)
builder.add_edge(START, "think")
builder.add_conditional_edges("think", route_after_think,
                              {"act": "act", "reprompt": "reprompt", "answer": "answer", "give_up": "give_up"})
builder.add_edge("act", "think")
builder.add_edge("reprompt", "think")
builder.add_edge("answer", END)
builder.add_edge("give_up", END)
react_graph = builder.compile()


def run_agent(question: str, history: list[dict] | None = None, persona: str = DEFAULT_PERSONA,
              variant: dict | None = None, verbose: bool = True) -> dict:
    """Run one user message through the ReAct graph and return the final state."""
    v = {"model": AGENT_MODEL, **(variant or {})}
    if verbose:
        print(f"\n[USER] {question}\n[PERSONA] {PERSONAS[persona]['name']} | model {v['model']}")
    state = react_graph.invoke(
        {"question": question, "history": history or [], "persona": persona, "variant": v, "transcript": [],
         "steps": [], "llm_calls": 0, "format_errors": 0, "latency": 0.0, "verbose": verbose},
        config={"recursion_limit": 4 * MAX_STEPS + 6},
    )
    state["tools_used"] = [s["tool"] for s in state.get("steps", [])]
    return state

# %% [gradio]
import gradio as gr

PERSONA_CHOICES = {p["name"]: key for key, p in PERSONAS.items()}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text(c) for c in content)
    if isinstance(content, dict):
        return content.get("text") or ""
    return str(content or "")


def to_history(history: list) -> list[dict]:
    """Gradio messages -> model history (user/model text only, trace boxes dropped)."""
    out = []
    for m in history or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        meta = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
        text = _text(m.get("content") if isinstance(m, dict) else getattr(m, "content", "")).strip()
        if role in ("user", "assistant") and not meta.get("title") and text:
            out.append({"role": "user" if role == "user" else "model", "text": text})
    return out[-8:]


def respond(message, history, persona_name):
    persona = PERSONA_CHOICES.get(persona_name, DEFAULT_PERSONA)
    state = None
    for model in [AGENT_MODEL] + APP_FALLBACK_MODELS:
        try:
            state = run_agent(_text(message), to_history(history), persona, {"model": model})
            break
        except (QuotaExhausted, ServiceUnavailable) as e:
            print(f"[UNAVAILABLE] {e}; trying the next model")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            return "Sorry, I'm having a technical hiccup. Please try again, or WhatsApp us at +961 70 000 111."
    if state is None:
        return "Sorry, today's usage limit is reached. Please try again later."

    cfg = {k: v for k, v in PERSONAS[persona]["config"].items() if k not in REJECTED_PARAMS}
    out = [gr.ChatMessage(content=", ".join(f"{k}={v}" for k, v in cfg.items()),
                          metadata={"title": f"⚙️ {PERSONAS[persona]['name']} · {state['variant']['model']}",
                                    "status": "done"})]
    for i, s in enumerate(state.get("steps", []), 1):
        body = (f"**Thought:** {s['thought']}\n\n**Action:** `{s['tool']}({json.dumps(s['args'], ensure_ascii=False)})`"
                f"\n\n**Observation:**\n```\n{s['observation'][:1200]}\n```")
        out.append(gr.ChatMessage(content=body, metadata={"title": f"🔁 Step {i}: {s['tool']}", "status": "done"}))
    last_thought = state.get("parsed", {}).get("thought")
    if last_thought:
        out.append(gr.ChatMessage(content=f"**Thought:** {last_thought}",
                                  metadata={"title": "💭 Final thought", "status": "done"}))
    out.append(gr.ChatMessage(content=state["final_answer"]))
    return out


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[gr.Dropdown(list(PERSONA_CHOICES), value=PERSONAS[DEFAULT_PERSONA]["name"], label="Persona")],
    additional_inputs_accordion=gr.Accordion("Persona", open=True),
    title="☀️ SunCedar Solar · ReAct agent with personas",
    description="Pick a persona, then ask about packages, prices, warranty or service areas, describe your home for a "
                "recommendation, book a site visit, or leave feedback. Each reply shows its Thought → Action → "
                "Observation steps. (SunCedar Solar is a fictional company.)",
    examples=[["How much is the Family package and what's included?", PERSONAS["nour"]["name"]],
              ["We're 6 people with 2 ACs and a water pump. Which package?", PERSONAS["karim"]["name"]],
              ["Book me a site visit for the Villa. I'm Lara Haddad, lara@example.com", PERSONAS["salma"]["name"]],
              ["Kifak! adde se3er l Basic package?", PERSONAS["nour"]["name"]]],
    cache_examples=False,  # Spaces caches examples by default, which would call the API at startup
)

if __name__ == "__main__":
    demo.launch()
