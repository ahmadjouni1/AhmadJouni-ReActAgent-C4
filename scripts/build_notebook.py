"""Build business_react_agent.ipynb from the sections of app.py and experiments.py plus notebook-only cells.

app.py and experiments.py are the single source of truth: every `# %% [name]` section becomes one code cell,
so the notebook, the deployed app and `python experiments.py` all run the same code.

Usage:
    python scripts/build_notebook.py
    python -m nbconvert --to notebook --execute --inplace business_react_agent.ipynb
"""

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = {}
for source in ("app.py", "experiments.py"):
    text = (ROOT / source).read_text(encoding="utf-8")
    for m in re.finditer(r"^# %% \[(\w+)\]\n(.*?)(?=^# %% \[|\Z)", text, re.DOTALL | re.MULTILINE):
        SECTIONS[m.group(1)] = m.group(2).strip("\n")
SECTIONS.pop("main")  # the CLI entry point of experiments.py; the notebook has its own cells for it


def src(name):
    return new_code_cell(SECTIONS.pop(name))


md, code = new_markdown_cell, new_code_cell
REFLECTION = (ROOT / "scripts" / "reflection.md").read_text(encoding="utf-8")

cells = [
    md("""# ☀️ SunCedar Solar · Custom ReAct Agent with Configurable Personas
**EECE 503P/798S · C4: Agent Personas & Thinking Mechanisms** · Ahmad Jouni

This notebook turns the C3 SunCedar Solar business chatbot into a **ReAct agent** that reasons and acts through tools.
The ReAct loop is written by hand as a **LangGraph `StateGraph`** (no prebuilt agent or executor). The model writes
`Thought / Action / Final Answer` as plain text; our code parses the Action, runs the tool, feeds back an
`Observation`, and loops. Three **personas** (prompt traits + their own LLM configuration) share one **instructions
block** and four **tools**, and a logged experiment grid compares personas, LLM settings and prompt-engineering choices.

```
                    ┌────────────────────────── LangGraph StateGraph ──────────────────────────┐
 user message ──►   │  think (LLM) ──► parse ──┬─ Action ─────► act (run tool) ──► Observation ─┐ │
 + persona          │       ▲                  │                                                │ │
 + chat history     │       └──────────────────┼──────────────────────────────────────────────── ┘ │
                    │                          ├─ malformed ──► reprompt (format reminder, once)  │
                    │                          ├─ Final Answer ──► answer ──► END                 │
                    │                          └─ max steps / still malformed ──► give_up ──► END │
                    └──────────────────────────────────────────────────────────────────────────────┘
 tools: search_docs (BM25 over me/ docs) · recommend_package · record_customer_interest · record_feedback
```

| § | Section | Assignment item |
|---|---|---|
| 1 | Use case | 2.1 use-case summary |
| 2 | LLM setup & parameter support | 2.3 configurations (what the API actually honors) |
| 3 | Knowledge base & tools | 2.1 tools |
| 4 | Personas | 2.3 personas = prompt traits + LLM config |
| 5 | Instructions block | 2.4 clear instructions |
| 6 | ReAct loop in LangGraph | 2.2 manual ReAct logic |
| 7–8 | Experiments & results | 2.3 experiments, logged and compared |
| 9 | Reflection | 2.5 reflection questions |
| 10 | Gradio app | working assistant |

> Code cells are generated from `app.py` and `experiments.py` (see `scripts/build_notebook.py`). Every LLM response is
> cached in `cache/llm_cache.jsonl`, so re-running this notebook reproduces the results without spending API quota."""),

    md("""## 0 · Setup
**Locally:** `pip install -r requirements.txt`, then put your key in `.env` as `GEMINI_API_KEY=...`.
**Google Colab:** add `GEMINI_API_KEY` in the 🔑 *Secrets* panel (enable *Notebook access*); the next cell clones the
repo (documents, cache and results), installs the requirements and loads the key. Locally it does nothing."""),
    code("""# Google Colab only: fetch the project files, install dependencies, load the key from Colab Secrets.
import os, sys
if "google.colab" in sys.modules:
    if not os.path.exists("me"):
        if os.path.exists("AhmadJouni-ReActAgent-C4"):
            !git -C AhmadJouni-ReActAgent-C4 pull -q
        else:
            !git clone -q https://github.com/ahmadjouni1/AhmadJouni-ReActAgent-C4.git
        %cd AhmadJouni-ReActAgent-C4
    %pip install -q -r requirements.txt
    from google.colab import userdata
    os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
    print("Colab setup done:", os.getcwd())"""),
    src("setup"),

    md("""## 1 · Use case
**SunCedar Solar** (fictional, reused from C3) designs and installs home solar systems in Lebanon.

| | |
|---|---|
| **Agent goal** | Help visitors understand SunCedar's packages, prices, warranty, payment options and service areas; recommend the right package for their home; capture leads (name + email) for a free site visit; log complaints and questions the documents can't answer. |
| **Target audience** | Lebanese homeowners and families tired of short EDL (state grid) hours and expensive generator subscriptions (*ishtirak*), writing in English, Arabic or Arabizi. Mostly non-technical; some want exact specs. |
| **Tools** | `search_docs` (BM25 search over the company profile PDF + summary), `recommend_package` (deterministic sizing calculator), `record_customer_interest` (lead logging), `record_feedback` (unanswered questions & complaints). |
| **Constraints** | Facts only from `me/about_business.pdf` and `me/business_summary.txt`, retrieved through `search_docs` (the documents are **not** pasted into the prompt); never invent prices, specs, discounts or policies; leads need both name and email; replies in the user's language. |
| **Limitations** | Keyword search misses synonyms and Arabizi terms (the agent must translate queries to English keywords); `recommend_package` uses simple rules, the final price is set after a site visit; no real booking system; free-tier quota (Flash-Lite 500 requests/day, Flash 20/day); temperature/top-p only work on some models (§2). |"""),

    md("""## 2 · LLM setup and parameter support
We use Gemini through the **native `google-genai` SDK**, so every sampling parameter can be set. `generate()` adds three
things around `client.models.generate_content`: a **disk cache** keyed by model + system prompt + messages + config,
**rate limiting** per model (free-tier RPM), and retries on 429/503. Parameters the API rejects are dropped with a warning."""),
    src("llm"),
    md("""### 2.1 Which parameters do the models actually honor?
Accepting a parameter is not the same as using it. For each model we sample the same prompt **3 times** per setting:
if `temperature=0` still gives 3 different sentences, the parameter is ignored. (`top_p`/`top_k` are tested at
temperature 1.8, where only they can make the output repeatable.) Results are cached, so this costs nothing on re-runs."""),
    src("probe"),
    code("""import pandas as pd
probe = pd.DataFrame([probe_parameters(m) for m in [AGENT_MODEL, "gemini-3.5-flash-lite"]]).set_index("model").T
probe"""),
    md("""**What this means for the personas**

| Parameter | `gemini-3.1-flash-lite` (agent) | `gemini-3.5-flash-lite` | `gemini-3.6/3.8-flash` |
|---|---|---|---|
| temperature | ✅ honored | ⚠️ accepted but ignored | ignored (Google docs: "Gemini 3.6+ models ignore custom values for temperature, top_p, top_k") |
| top_p | ✅ honored | see table above | ignored (docs) |
| top_k | ⚠️ accepted, no measurable effect | ⚠️ accepted, no effect | ignored (docs) |
| max_output_tokens · stop_sequences · seed | ✅ | ✅ | ✅ |
| presence_penalty · frequency_penalty | ❌ HTTP 400 *"Penalty is not enabled for this model"* | ❌ | ❌ (also Gemma 4) |

So the agent runs on **`gemini-3.1-flash-lite`**: it is the current model where temperature and top-p really change
the output, and it has the largest free quota (500 requests/day). The penalties stay in the persona configs to document
the intent, but `build_config()` drops them; varied phrasing for Nour is requested in her prompt instead."""),

    md("""## 3 · Knowledge base and tools
The two C3 documents are split into **24 passages** (summary paragraphs + overlapping PDF windows, each prefixed with
its page heading) and indexed with **BM25**. The agent never sees the whole documents: it must call `search_docs`.
The tools return plain text or JSON, which becomes the `Observation`."""),
    src("knowledge"),
    src("tools"),
    md("**Tools called directly (no LLM):** what the agent sees as Observations."),
    code("""print(search_docs("battery warranty"), "\\n")
print(recommend_package(5, ["fridge", "water pump", "washing machine", "1 AC"]), "\\n")
print(execute_tool("book_flight", {}))
print("\\nTool descriptions embedded in the system prompt:\\n" + TOOLS_PROMPT)"""),

    md("""## 4 · Personas
Each persona is defined by **prompt traits** (tone, vocabulary, response style, values) **and** a matching **LLM
configuration**. The rationale links each setting to the persona it supports."""),
    src("personas"),
    code("""rows = []
for key, p in PERSONAS.items():
    for setting, why in p["rationale"].items():
        rows.append({"persona": p["name"], "setting": setting, "why": why})
pd.set_option("display.max_colwidth", 120)
pd.DataFrame(rows).set_index(["persona", "setting"])"""),

    md("""## 5 · Instructions block
One **instructions block is shared by all personas** (personas change *how* the agent sounds; instructions say *what*
it must do), written to the guidelines from class:
- **one purpose per rule** (13 numbered rules, none bundles two goals);
- **explicit input and output**: what arrives each turn, and the exact ReAct output format;
- **a complete worked example** (Thought → Action → Observation → Final Answer) inside the instructions;
- **no conflicts with any persona**: no rule dictates tone or length (that is the persona's job), and persona traits
  never contradict a rule (e.g. Karim's "at most 60 words" fits rule 11's language rule; Salma's read-back happens
  *after* rule 10 saves the lead).

The variants used in the prompt-engineering experiments are built from the same parts: without Chain-of-Thought
(no `Thought:` lines), zero-shot (no example) and few-shot (3 extra examples)."""),
    src("instructions"),
    code("print(INSTRUCTIONS)"),
    code("""print(build_system_prompt("karim")[:1400], "\\n...")"""),

    md("""## 6 · The ReAct loop in LangGraph (manual)
**Detecting the action** is done by our parser, not by the model's function-calling API:
- `Action: tool_name({json})` → run the tool (JSON, `key="value"` or a single positional argument are all accepted);
- `Final Answer: ...` → done;
- anything else → **malformed**: the graph re-prompts **once** with a format reminder, then gives up gracefully;
- an unknown tool or bad arguments come back as an `Observation: Error ...` so the model can correct itself;
- `stop_sequences=["Observation:"]` stops the model before it can invent an Observation;
- at most `MAX_STEPS = 6` model calls per message."""),
    src("parser"),
    code("""for text in ['Thought: I need the price.\\nAction: search_docs({"query": "Family price"})',
             'Action: search_docs(query="battery warranty")',
             'Thought: Done.\\nFinal Answer: The Family package is $6,450.',
             'Sure! It costs about 6000 dollars.']:
    print(f"{text!r:75} -> {parse_react(text)}")"""),
    md("""The graph: `think` is the LLM node, `act` the tool node, and the **conditional edge** `route_after_think` reads the
parsed output to choose the next node. State carries the question, persona, history, the growing transcript of
model steps and Observations, and counters."""),
    src("graph"),
    code("""from IPython.display import Image, display
try:
    display(Image(react_graph.get_graph().draw_mermaid_png()))
except Exception as e:  # rendering uses the mermaid.ink web service; fall back to the saved diagram / text
    print(f"(live rendering unavailable: {type(e).__name__})")
    if (ROOT / "docs" / "react_graph.png").exists():
        display(Image(str(ROOT / "docs" / "react_graph.png")))
    else:
        print(react_graph.get_graph().draw_mermaid())"""),
    md("""### 6.1 Full traces, one per persona
Multi-step examples: Salma recommends a package **and** looks up the installment plan (two different tools before the
Final Answer); Karim answers a spec question; Nour handles a lead in Arabizi over two turns (chat history = short-term memory)."""),
    code("""set_log_dir(ROOT / "experiments" / "demo_logs")   # keep demo leads out of the live app's logs
s1 = run_agent("We're 6 people with 2 ACs and a water pump. Which package do you recommend, and can I pay in installments?",
               persona="salma")
s2 = run_agent("What inverter and battery does the Villa package use, and what is the battery warranty?", persona="karim")"""),
    code("""history = []
for msg in ["Ahla! Baddi a3mol site visit lal Family package.",
            "Ana Rami Saad, rami.saad@example.com, men Saida."]:
    s = run_agent(msg, history, persona="nour")
    history += [{"role": "user", "text": msg}, {"role": "model", "text": s["final_answer"]}]
set_log_dir(LOG_DIR)"""),

    md("""## 7 · Experiments
**Test set:** 6 fixed queries, one per behaviour (price, package recommendation, lead with name + email, unanswerable
question, complaint, Arabizi), each with the tool(s) it should use, key facts, and the expected behaviour for the judge.

**Grid** (one factor at a time against a control, not a full cartesian product; 15 variants × 6 queries = 90 runs):
- **A. Personas:** Nour, Karim, Salma, each with its own prompt and config;
- **B. LLM configuration** on the control persona (Salma): temperature 0.0 / 1.3, top_p 0.3, top_k 1, max_output_tokens
  120, thinking_level low, and **model type** (`gemini-3.5-flash-lite`, `gemini-3.8-flash`); the penalties cannot be
  varied (rejected by the API, §2.1);
- **C. Prompt engineering** on the control persona: without Chain-of-Thought, zero-shot, few-shot, instructions in
  the user message instead of the system prompt.

**Automatic scores per run:** correct tool used · valid ReAct format (no re-prompt needed) · no invented price (every
`$` amount must appear in the documents) · key-fact recall · number of LLM calls · latency · and an **LLM judge**
(`gemini-3.5-flash-lite`, temperature 0) for helpfulness, persona fit and same language. `quality` = mean of tool,
format, price, facts and rescaled helpfulness (0–1)."""),
    src("testset"),
    src("grid"),
    src("scoring"),
    src("runner"),
    code("""results = run_experiments()   # cached: re-runs are free; an interrupted run resumes from the cache"""),

    md("## 8 · Results"),
    src("summary"),
    md("### 8.1 Personas"),
    code("summarize(results, 'persona')"),
    src("charts"),
    code("plot_personas(results); plt.show()"),
    md("### 8.2 LLM configuration (control persona: Salma)"),
    code("summarize(results, 'config')"),
    code("plot_variants(results, 'config', 'LLM configuration (control persona: Salma)'); plt.show()"),
    md("### 8.3 Prompt engineering (control persona: Salma)"),
    code("summarize(results, 'prompt')"),
    code("plot_variants(results, 'prompt', 'Prompt engineering (control persona: Salma)'); plt.show()"),
    md("### 8.4 Per-query breakdown and failures"),
    code("""results.pivot_table(index="variant", columns="case", values="quality", sort=False).round(2)"""),
    code("""fails = results[~(results.tool_ok & results.format_ok & results.price_ok)]
fails[["variant", "case", "tools_used", "tool_ok", "format_ok", "price_ok", "unsupported_prices", "answer"]]"""),
    md("### 8.5 Same question, three personas"),
    code("""for _, r in results[(results.exp == "persona") & (results.case == "arabizi")].iterrows():
    print(f"--- {r.variant}  (tools: {r.tools_used}; judge helpfulness {r.judge_helpfulness}, persona fit "
          f"{r.judge_persona_fit}, same language {r.judge_same_language})\\n{r.answer}\\n")"""),

    md(REFLECTION),

    md("""## 10 · Gradio app
A `ChatInterface` with a **persona dropdown**. Every reply shows the configuration used and each ReAct step
(Thought, Action, Observation) in collapsible boxes, then the Final Answer. The same code is `app.py` (Hugging Face
Spaces ready)."""),
    src("gradio"),
]

assert not SECTIONS, f"sections not placed in the notebook: {list(SECTIONS)}"
nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}})
out = ROOT / "business_react_agent.ipynb"
nbformat.write(nb, out)
print(f"Wrote {out} ({len(cells)} cells)")
