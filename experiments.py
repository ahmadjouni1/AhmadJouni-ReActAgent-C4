"""Experiments for the SunCedar ReAct agent: parameter-support probe, persona / configuration / prompt-engineering
comparisons, automatic scoring and an LLM judge. Every LLM response is cached on disk (cache/llm_cache.jsonl),
so re-running costs no quota and an interrupted run resumes where it stopped.

Usage:  python experiments.py      (or run the experiment sections of business_react_agent.ipynb)
"""
from app import *  # noqa: F401,F403  (the notebook runs these sections in the same namespace as app.py)
from app import _cache, _store, _throttle

# %% [probe]
PROBE_PROMPT = [{"role": "user", "text": "Write one short creative sentence about the sun over Beirut."}]


def _samples(model: str, config: dict, n: int = 3) -> list[str]:
    """n samples of the same prompt + config (each in its own cache slot)."""
    return [generate(model, None, PROBE_PROMPT, config, sample=i)["text"].strip() for i in range(n)]


def probe_parameters(model: str) -> dict:
    """Empirically check which parameters a model accepts and whether they change the output."""
    row = {"model": model}
    hot = _samples(model, {"temperature": 1.8})
    cold = _samples(model, {"temperature": 0.0})
    row["temperature"] = (f"honored (T=0: {len(set(cold))}/3 distinct, T=1.8: {len(set(hot))}/3)"
                          if len(set(cold)) == 1 < len(set(hot)) else
                          f"accepted, IGNORED (T=0 still {len(set(cold))}/3 distinct)")
    tp = _samples(model, {"temperature": 1.8, "top_p": 0.01})
    row["top_p"] = (f"honored (top_p=0.01 at T=1.8: {len(set(tp))}/3 distinct)" if len(set(tp)) == 1
                    else f"accepted, no effect ({len(set(tp))}/3 distinct)")
    tk = _samples(model, {"temperature": 1.8, "top_k": 1})
    row["top_k"] = (f"honored (top_k=1 at T=1.8: {len(set(tk))}/3 distinct)" if len(set(tk)) == 1
                    else f"accepted, no effect (top_k=1 at T=1.8: {len(set(tk))}/3 distinct)")
    mt = generate(model, None, PROBE_PROMPT, {"max_output_tokens": 8})
    row["max_output_tokens"] = f"honored (limit 8 -> {mt['usage']['out']} tokens, finish={mt['finish']})"
    st = generate(model, None, PROBE_PROMPT, {}, stop=["Beirut"])
    row["stop_sequences"] = "honored" if "Beirut" not in st["text"] else "ignored"
    for p in ("presence_penalty", "frequency_penalty"):
        # Raw API call (build_config() would drop the penalty), cached like every other call.
        key = f"penalty-probe:{model}:{p}"
        if key not in _cache:
            _throttle(model)
            try:
                client.models.generate_content(
                    model=model, contents=PROBE_PROMPT[0]["text"], config=types.GenerateContentConfig(**{p: 0.5}))
                result = "accepted"
            except genai_errors.ClientError as e:
                result = f"REJECTED (HTTP {e.code}: {e.message})"
            _store({"key": key, "model": model, "result": result})
        row[p] = _cache[key]["result"]
    return row

# %% [testset]
# Fixed test set: one query per behaviour the agent must handle, with the tool(s) it should use and key facts.
TEST_SET = [
    {"id": "price", "query": "How much is the Family package, and what's included in the price?",
     "expected_tools": [["search_docs"]], "facts": ["6,450"],
     "behaviour": "States the Family price ($6,450) and what is included, from the documents."},
    {"id": "recommend", "query": "We're a family of 5 with a fridge, a water pump, a washing machine and one AC. "
                                 "Which package should we get?",
     "expected_tools": [["recommend_package"]], "facts": ["Family", "6,450"],
     "behaviour": "Recommends the Family package ($6,450) using the sizing tool, and invites a site visit."},
    {"id": "lead", "query": "Please book a free site visit for the Villa package. I'm Ahmad Jouni, "
                            "ahmad.jouni@example.com, I live in Batroun.",
     "expected_tools": [["record_customer_interest"]], "facts": ["ahmad.jouni@example.com"],
     "behaviour": "Saves the lead (name + email + Villa) and confirms the details back to the user."},
    {"id": "unknown", "query": "Do you install EV chargers for electric cars?",
     "expected_tools": [["record_feedback"]], "facts": [],
     "behaviour": "Does not invent an answer; logs the question for the team and says they will follow up."},
    {"id": "complaint", "query": "Your technician came 3 hours late to my maintenance visit and didn't even apologize.",
     "expected_tools": [["record_feedback"]], "facts": [],
     "behaviour": "Apologizes, logs the complaint for the team and says they will follow up."},
    {"id": "arabizi", "query": "Kifak! adde se3er l Basic package w shu fi battery?",
     "expected_tools": [["search_docs"], ["recommend_package"]], "facts": ["3,450", "5.12"],
     "behaviour": "Replies in Arabizi with the Basic price ($3,450) and its 5.12 kWh LiFePO4 battery."},
]
print(f"Test set: {len(TEST_SET)} queries -> {', '.join(c['id'] for c in TEST_SET)}")

# %% [grid]
# A deliberately small grid: one factor changes at a time against a control (not a full cartesian product).
CONTROL = "salma"  # the Cautious Helper with moderate settings is the control for the config and prompt experiments


def spec(exp, name, persona, **variant):
    return {"exp": exp, "name": name, "persona": persona, "variant": variant}


VARIANTS = [
    # A. Personas: prompt traits + their own LLM configuration
    spec("persona", "nour (friendly, T=0.8)", "nour"),
    spec("persona", "karim (strict, T=0.1)", "karim"),
    spec("persona", "salma (cautious, T=0.3)", "salma"),
    # B. LLM configuration: one parameter at a time on the control persona
    spec("config", "temperature 0.0", CONTROL, config={"temperature": 0.0}),
    spec("config", "temperature 1.3", CONTROL, config={"temperature": 1.3}),
    spec("config", "top_p 0.3", CONTROL, config={"top_p": 0.3}),
    spec("config", "top_k 1", CONTROL, config={"top_k": 1}),
    spec("config", "max_output_tokens 120", CONTROL, config={"max_output_tokens": 120}),
    spec("config", "thinking_level low", CONTROL, config={"thinking_level": "low"}),
    spec("config", "model gemini-3.5-flash-lite", CONTROL, model="gemini-3.5-flash-lite"),
    spec("config", "model gemini-3.8-flash", CONTROL, model="gemini-3.8-flash"),
    # C. Prompt engineering on the control persona
    spec("prompt", "no Chain-of-Thought", CONTROL, cot=False),
    spec("prompt", "zero-shot (no example)", CONTROL, shots="zero"),
    spec("prompt", "few-shot (4 examples)", CONTROL, shots="few"),
    spec("prompt", "instructions in user message", CONTROL, placement="message"),
    spec("prompt", "no rules (format + example only)", CONTROL, rules=False),
]
print(f"Grid: {len(VARIANTS)} variants x {len(TEST_SET)} queries = {len(VARIANTS) * len(TEST_SET)} runs")

# %% [scoring]
EXP_DIR = ROOT / "experiments"
PRICE_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(?:USD|dollars?)\b", re.I)

JUDGE_PROMPT = """You grade replies from a solar company's chatbot. Return JSON only:
{{"helpfulness": 1-5, "persona_fit": 1-5, "same_language": true|false, "comment": "<one short sentence>"}}
- helpfulness: does the reply do what the expected behaviour describes, correctly and clearly? (5 = fully)
- persona_fit: does the tone, vocabulary and style match the persona description? (5 = perfectly)
- same_language: is the reply in the same language/script as the user (English, Arabic, or Arabizi)?

PERSONA: {persona}
USER: {question}
EXPECTED BEHAVIOUR: {behaviour}
REPLY: {reply}"""


def _norm(s: str) -> str:
    return re.sub(r"[,\s]", "", s.lower())


def unsupported_prices(answer: str) -> list[int]:
    """Dollar amounts in the answer that never appear in the documents (= invented or computed prices)."""
    found = []
    for m in PRICE_RE.finditer(answer or ""):
        value = float((m.group(1) or m.group(2)).replace(",", ""))
        if value >= 10 and int(value) not in DOC_PRICES:
            found.append(int(value))
    return found


def judge(case: dict, persona: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(persona=PERSONAS[persona]["prompt"], question=case["query"],
                                 behaviour=case["behaviour"], reply=answer)
    try:
        out = generate(JUDGE_MODEL, None, [{"role": "user", "text": prompt}],
                       {"temperature": 0.0, "response_mime_type": "application/json"})
        data = json.loads(re.search(r"\{.*\}", out["text"], re.S).group(0))
        return {"judge_helpfulness": int(data.get("helpfulness", 0)), "judge_persona_fit": int(data.get("persona_fit", 0)),
                "judge_same_language": bool(data.get("same_language")), "judge_comment": data.get("comment", "")}
    except (QuotaExhausted, ValueError, AttributeError, json.JSONDecodeError) as e:
        return {"judge_helpfulness": None, "judge_persona_fit": None, "judge_same_language": None,
                "judge_comment": f"judge failed: {type(e).__name__}"}


def score(case: dict, variant: dict, state: dict) -> dict:
    answer = state.get("final_answer", "")
    used = state.get("tools_used", [])
    tool_ok = any(all(t in used for t in option) for option in case["expected_tools"])
    facts = [f for f in case["facts"] if _norm(f) in _norm(answer)]
    bad_prices = unsupported_prices(answer)
    row = {
        "exp": variant["exp"], "variant": variant["name"], "persona": variant["persona"],
        "model": state["variant"]["model"], "case": case["id"],
        "tool_ok": tool_ok, "format_ok": state.get("status") == "answered" and state.get("format_errors", 0) == 0,
        "price_ok": not bad_prices, "fact_recall": len(facts) / len(case["facts"]) if case["facts"] else 1.0,
        "status": state.get("status"), "format_errors": state.get("format_errors", 0),
        "tool_calls": len(used), "llm_calls": state.get("llm_calls", 0),
        "latency_s": round(state.get("latency", 0.0), 2), "answer_words": len(answer.split()),
        "tools_used": " > ".join(used), "unsupported_prices": bad_prices, "answer": answer,
    }
    row.update(judge(case, variant["persona"], answer))
    parts = [row["tool_ok"], row["format_ok"], row["price_ok"], row["fact_recall"]]
    if row["judge_helpfulness"]:
        parts.append((row["judge_helpfulness"] - 1) / 4)
    row["quality"] = round(sum(map(float, parts)) / len(parts), 3)
    return row

# %% [runner]
import pandas as pd


def run_experiments(variants: list[dict] = VARIANTS, cases: list[dict] = TEST_SET) -> pd.DataFrame:
    """Run every (variant, query) pair through the ReAct graph and score it. Cached calls are free, so an
    interrupted run (e.g. daily quota) simply resumes from the cache next time."""
    tool_logs = EXP_DIR / "tool_logs"
    for f in ("leads.jsonl", "feedback.jsonl"):        # tools log to their own folder during experiments
        (tool_logs / f).unlink(missing_ok=True)
    set_log_dir(tool_logs)
    rows, traces = [], []
    t_start, calls_before = time.time(), len(_cache)
    try:
        for v in variants:
            for case in cases:
                try:
                    state = run_agent(case["query"], [], v["persona"], v["variant"], verbose=False)
                except (QuotaExhausted, ServiceUnavailable) as e:
                    print(f"[STOPPED] {e}. Re-run this cell later (after the daily reset for quota): "
                          "finished runs come back from the cache for free.")
                    raise StopIteration
                row = score(case, v, state)
                rows.append(row)
                traces.append({**{k: row[k] for k in ("exp", "variant", "case")}, "query": case["query"],
                               "steps": state.get("steps", []), "final_answer": row["answer"]})
                print(f"[{v['exp']:<7}] {v['name']:<30} {case['id']:<9} tools={row['tools_used'] or '-':<45} "
                      f"tool={'Y' if row['tool_ok'] else 'N'} fmt={'Y' if row['format_ok'] else 'N'} "
                      f"price={'Y' if row['price_ok'] else 'N'} facts={row['fact_recall']:.0%} "
                      f"judge={row['judge_helpfulness']}/{row['judge_persona_fit']} q={row['quality']:.2f}")
    except StopIteration:
        pass
    finally:
        set_log_dir(LOG_DIR)
    df = pd.DataFrame(rows)
    EXP_DIR.mkdir(exist_ok=True)
    df.to_csv(EXP_DIR / "results.csv", index=False)
    with (EXP_DIR / "traces.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n{len(df)} runs saved to experiments/results.csv (+ traces.jsonl) in {time.time() - t_start:.0f}s; "
          f"{len(_cache) - calls_before} new API calls")
    return df

# %% [summary]
METRICS = {"tool_ok": "tool accuracy", "format_ok": "valid ReAct format", "price_ok": "no invented price",
           "fact_recall": "key facts", "judge_helpfulness": "helpfulness (1-5)", "judge_persona_fit": "persona fit (1-5)",
           "judge_same_language": "same language", "llm_calls": "LLM calls", "latency_s": "latency (s)",
           "quality": "quality (0-1)"}


def summarize(df: pd.DataFrame, exp: str, with_control: bool = True) -> pd.DataFrame:
    """Mean of each metric per variant; config/prompt tables include the control persona's row as baseline."""
    part = df[df.exp == exp]
    if with_control and exp != "persona":
        control = df[(df.exp == "persona") & (df.persona == CONTROL)].assign(variant="control (salma defaults)")
        part = pd.concat([control, part])
    numeric = part[["variant"]].join(part[list(METRICS)].astype(float))
    table = numeric.groupby("variant", sort=False).mean().rename(columns=METRICS)
    return table.round(2)

# %% [charts]
import matplotlib.pyplot as plt

# Reference categorical palette (fixed order, validated for color-vision deficiency), light surface.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # slots 1-3: blue, orange, aqua
NEUTRAL = "#a3a29c"                          # control / baseline rows
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK, "font.size": 10,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11})


def _clean(ax, xmax=None):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xmax:
        ax.set_xlim(0, xmax)


def plot_personas(df: pd.DataFrame):
    """Grouped bars: each persona's score on the 0-1 metrics (judge scores rescaled from 1-5)."""
    t = summarize(df, "persona")
    cols = {"tool accuracy": 1, "valid ReAct format": 1, "no invented price": 1, "key facts": 1,
            "helpfulness (1-5)": 5, "persona fit (1-5)": 5, "same language": 1}
    data = pd.DataFrame({c: (t[c] - 1) / 4 if scale == 5 else t[c] for c, scale in cols.items()})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    n, h = len(data), 0.26
    for i, (variant, row) in enumerate(data.iterrows()):
        y = [k + (i - (n - 1) / 2) * (h + 0.02) for k in range(len(cols))]
        ax.barh(y, row.values, height=h, color=SERIES[i], label=variant, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(cols)), [c.replace(" (1-5)", " (judge)") for c in cols])
    ax.invert_yaxis()
    _clean(ax, 1.05)
    ax.set_xlabel("score (0-1; judge scores rescaled from 1-5)")
    ax.set_title("Persona comparison: same 6 queries, each persona with its own prompt + LLM config", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(EXP_DIR / "chart_personas.png", dpi=150)
    return fig


def plot_variants(df: pd.DataFrame, exp: str, title: str):
    """Two panels sharing the variant axis: overall quality, and latency (no dual axis)."""
    t = summarize(df, exp)
    colors = [NEUTRAL if v.startswith("control") else SERIES[0] for v in t.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.42 * len(t) + 1.4), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 2]})
    for ax, col, xmax, fmt in ((a1, "quality (0-1)", 1.1, "{:.2f}"), (a2, "latency (s)", None, "{:.1f}s")):
        ax.barh(range(len(t)), t[col], height=0.6, color=colors, edgecolor=SURFACE, linewidth=1)
        for y, v in enumerate(t[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8.5, color=INK_2)
        _clean(ax, xmax if xmax else t[col].max() * 1.25)
        ax.set_xlabel(col)
    a1.set_yticks(range(len(t)), t.index)
    a1.invert_yaxis()
    a1.set_title(title, loc="left")
    a2.set_title("model latency per query (noisy: API overload)", loc="left", fontweight="normal", color=INK_2)
    fig.tight_layout()
    fig.savefig(EXP_DIR / f"chart_{exp}.png", dpi=150)
    return fig

# %% [main]
if __name__ == "__main__":
    for m in [AGENT_MODEL, "gemini-3.5-flash-lite"]:
        print(probe_parameters(m))
    results = run_experiments()
    for exp in ("persona", "config", "prompt"):
        print(f"\n=== {exp}\n", summarize(results, exp).to_string())
    plot_personas(results)
    plot_variants(results, "config", "LLM configuration (control persona: Salma)")
    plot_variants(results, "prompt", "Prompt engineering (control persona: Salma)")
