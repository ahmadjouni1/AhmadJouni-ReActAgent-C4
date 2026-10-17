# Demo video script

Run `python app.py`, open http://127.0.0.1:7860, and keep the terminal visible next to the browser.
Pick the persona in the **Persona** dropdown under the chat. Every reply shows:
- **⚙️ persona · model**: the LLM configuration used;
- **🔁 Step N: tool**: each ReAct step, containing that step's Thought, Action and Observation;
- **💭 Final thought**, then the Final Answer.

The terminal prints the same trace (`Thought:`, `Action:`, `Observation:`, `Final Answer:`) plus `[TOOL]` log lines.
Use the chat's **Clear** button (🗑) between parts.

## Part 1: Nour (Friendly Advisor), single tool
Persona: **Nour – Friendly Advisor**
| Message | Expect |
|---|---|
| `Ahla! How much is the Family package and what's included?` | Step 1: `search_docs` → $6,450 and what's included, warm tone, invites a site visit |

## Part 2: Eng. Karim (Strict Expert), multi-step ReAct trace
Persona: **Eng. Karim – Strict Expert**
| Message | Expect |
|---|---|
| `We're 6 people with 2 ACs and a water pump. Which package, and what is the battery warranty?` | Step 1: `recommend_package` → Villa · Step 2: `search_docs` → 10 years / 6,000 cycles · terse bullet answer with specs |

## Part 3: Salma (Cautious Helper), lead capture
Persona: **Salma – Cautious Helper**
| Message | Expect |
|---|---|
| `I'd like a free site visit for the Villa package.` | No tool, or `search_docs` only; asks for name and email |
| `I'm Lara Haddad, lara@example.com, from Jbeil.` | `record_customer_interest` → terminal `[TOOL] ... LEAD:`; Salma reads the details back |

## Part 4: Unanswerable question, two-step trace
Any persona (Salma shows it best)
| Message | Expect |
|---|---|
| `Do you install EV chargers for electric cars?` | Step 1: `search_docs` (no match) → Step 2: `record_feedback` → "passed to the team" |

## Part 5: Complaint
| Message | Expect |
|---|---|
| `Your technician came 3 hours late and didn't apologize.` | `record_feedback`; apology; follow-up promised |

## Part 6: Arabizi
Persona: **Nour – Friendly Advisor**
| Message | Expect |
|---|---|
| `Kifak! adde se3er l Basic package w shu fi battery?` | `search_docs` with an **English** query (instruction rule 4) → Arabizi reply: $3,450, 5.12 kWh LiFePO4 |

## Part 7: Same question, three personas
Ask `How much is the Villa package?` three times, switching the dropdown each time (Clear in between). Show how
the same facts come out in three styles: Nour warm, Karim terse with specs, Salma citing the company profile.

## Part 8: Experiments
Open `business_react_agent.ipynb` (saved with outputs) and scroll through:
- §2.1 the parameter-support table (which parameters the models really honor);
- §6 the LangGraph diagram and the full traces;
- §8 the results tables and the three charts;
- §9 the reflection.

Or show `experiments/results.csv` and the charts `experiments/chart_*.png`.
