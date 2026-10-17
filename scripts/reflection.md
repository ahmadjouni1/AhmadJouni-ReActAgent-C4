<!-- DRAFT written from the actual experiment results: edit into your own words. -->
## 9 · Reflection

*Based on 96 logged runs (16 variants × 6 queries) in `experiments/results.csv`. Each cell is a single sample on a
6-query test set, so small differences (0.95 vs 0.99) are within noise; latency is not comparable across variants
because the API returned 503/504 "overloaded" errors during the runs.*

**Across all runs:** the right tool was used in **96.9%** of runs, the ReAct format was valid without a re-prompt in
**92.7%**, **no answer contained an invented price** (every `$` amount appeared in the documents), and no answer ever
mentioned a tool name or an Observation. The agent averaged 2.3 model calls per question; 22 runs chained two or more
tools.

### Which persona gave the most helpful or natural results?
- **Nour (Friendly Advisor, T=0.8)** was the most **natural**: the judge gave her 5.0/5 for both helpfulness and
  persona fit, and she was the only persona that answered the Arabizi question **in Arabizi**. Her weakness was the
  format: in 2 of 6 queries she wrote her reply in her own voice without the `Final Answer:` prefix (the warm persona at
  a higher temperature drifts out of the strict format). The one-time re-prompt fixed both cases. She also skipped
  reading the email back when saving the lead.
- **Eng. Karim (Strict Expert, T=0.1)** was the most **precise and concise** (37 words on average; 100% format, facts and
  tool use), but scored lowest with the judge (helpfulness 4.17, persona fit 3.67). His weakest case was the complaint
  (2/5 and 2/5): a terse, spec-first expert is a poor fit for an upset customer. He also dropped his bullet-point style
  on the lead query and answered the Arabizi question in English.
- **Salma (Cautious Helper, T=0.3)** was the most **helpful and reliable**: quality 1.00, perfect on every rule-based
  metric, helpfulness 5.0. The cost is length (83 words on average) and, like Karim, she answered the Arabizi question in
  English.

For SunCedar's audience, Salma is the safest default and Nour the most pleasant. Nour's language handling matters for
Lebanese users who write in Arabizi.

### Which prompt and configuration combination performed best?
**Salma's defaults on `gemini-3.1-flash-lite`**: instructions in the system prompt, Chain-of-Thought (`Thought:` lines),
one worked example, temperature 0.3, top_p 0.9. Quality 1.00 at the lowest steady latency (3.8 s per query).
- **Sampling parameters barely changed accuracy** on this tool-grounded task: temperature 0.0 and 1.3, top_p 0.3 and
  top_k 1 all scored 0.99, because the facts come from tool Observations, not from sampling. top_k having no effect
  matches the parameter probe (§2.1).
- **max_output_tokens 120** (0.95) still fit the answers but dropped the email read-back on the lead query.
  **thinking_level low** (0.96) added a format error and latency with no gain.
- **Model type had by far the largest effect.** `gemini-3.5-flash-lite` scored 0.80: only 2 of 6 runs had a valid
  format, it returned empty outputs (the graph gave up safely once), and it leaked Chinese characters into a reply.
  `gemini-3.8-flash` scored 0.99 but was the slowest (23.7 s), is capped at 20 free requests/day, and ignores
  temperature/top_p, so persona configurations would have no effect on it.
- **Prompt engineering:** zero-shot (1.00), few-shot (0.99) and no Chain-of-Thought (0.99) matched the control. On this
  model, the format instructions alone were enough. Few-shot slightly lowered persona fit (4.33 vs 4.50), likely because
  the examples' neutral voice leaked into the replies. **Moving the instructions into the user message (0.95) was the
  only prompt change that broke behaviour** (see below).

### How well did the agent reason and use tools?
Well on the core paths. The `Thought` lines matched the actions taken. The agent called `recommend_package` for
household descriptions and `search_docs` before every price. For questions the documents don't cover, it chained
`search_docs` → no match → `record_feedback`, and for leads it often checked the service area with `search_docs`
before `record_customer_interest`. Malformed outputs were caught by the parser (7 re-prompts in 96 runs, 1 give-up).
Two weaknesses:
1. After finding nothing on EV chargers, several personas said *"we don't currently offer EV chargers"*. The
   documents never say that, so it is a small invented fact that the judge did not penalize.
2. **It does not always notice an incomplete Observation.** In the §6.1 trace, Karim searched once for the Villa's
   inverter, battery and warranty. The passages covered the warranty but not the package table, and instead of
   searching again he answered "Hybrid inverter / LiFePO4", missing the 12 kW three-phase inverter and the 20.48 kWh
   battery.
3. **The language rule was the least reliable behaviour:** the reply matched the user's Arabizi in only 56% of runs
   (others answered in English or in Arabic script), even though the search queries were correctly translated to
   English keywords in 16 of 16 runs.

### Did the instructions block change the agent's behaviour? How?
Yes, mainly in **whether the agent actually does what it says**. In the ablation without the 13 rules (format and
example only), and again with the instructions moved into the user message, the agent searched for EV chargers, found
nothing, and replied *"I have logged your question"* **without calling `record_feedback`**: a false claim. With the
rules in the system prompt, the question was logged in every run on `gemini-3.1-flash-lite`. Other effects were
smaller than expected: prices stayed grounded and search queries stayed in English even without the rules, because the
output format and the tool descriptions already push the agent to search. The persona/instruction boundary also showed
a soft conflict. Rule 11 (reply in the user's language) lost to Karim's "technical vocabulary" and Salma's "plain
words" traits on the Arabizi query, so a persona's style wording can override a rule.

### What were the biggest challenges in implementation?
1. **What the API really honors.** Presence and frequency penalties are rejected (HTTP 400) by every current Gemini
   model, and `gemini-3.5-flash-lite` and the 3.6+/3.8 models ignore temperature. I had to measure this (§2.1) and
   choose `gemini-3.1-flash-lite` so that the persona configurations mean something.
2. **Keeping the text format robust.** Personas wrote replies without the `Final Answer:` prefix, one model returned
   empty outputs, arguments came as JSON, `key="value"` or a bare string, and models tend to invent their own
   `Observation:`. I handled these with a tolerant parser, a single format re-prompt, a give-up node and
   `stop_sequences=["Observation:"]`.
3. **Free-tier limits and overloads.** Flash allows 20 requests a day, and the API returned 503/504 errors during the
   runs. A disk cache (re-runs cost nothing), per-model throttling, retries and a resumable runner made the 96-run grid
   possible.
4. **Retrieval quality.** Plain BM25 ranked the maintenance-plan section above the package table for "Basic package
   price" until I added plural stemming and prefixed each chunk with its page heading.
5. **Evaluation.** The LLM judge was lenient (5/5 for a lead reply that never confirmed the email), and most variants hit
   the ceiling on a 6-query set. The rule-based checks (tool, format, price, facts) were more informative than the
   judge, and a larger test set with repeated samples would be the next step.
