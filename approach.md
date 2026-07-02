# Approach — SHL Assessment Recommendation Agent

## 1. Data: a pre-scraped JSON feed, reshaped into catalog.json

Rather than scraping `shl.com/products/product-catalog/` page by page, the
data comes from a single hosted JSON feed
(`https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json`)
that already has one flattened record per assessment. This is the same
underlying SHL catalog data the original scrape-based plan targeted, just
pre-flattened, so there's no HTML to parse and no pagination to walk.

`fetch_catalog.py` fetches that feed and reshapes each record into the
`catalog.json` schema `retrieval.py`'s `CatalogEntry` expects. This stays a
separate, inspectable offline step (`python fetch_catalog.py`) rather than
something baked into the Docker build, so the image stays reproducible
without needing network access to SHL's infrastructure at build time.
`catalog.json` remains the **only** data source the agent is allowed to
recommend from — everything downstream treats it as ground truth, never
the model's training-data knowledge of SHL's product line.

What the transform does, concretely:
- `duration` is free text (`"30 minutes"`, `""`, `"Variable"`, `"Untimed"`,
  `"0 minutes"`) parsed to an `int` minute count, or `None` when there's no
  fixed number — matching how `retrieval.py`'s `filter()` already treats
  `None` duration as passing any `max_duration_minutes` constraint.
- `remote` / `adaptive` are `"yes"`/`"no"` strings parsed to `bool`.
- The feed's free-text test-type labels are mapped to the single-letter
  codes (A/B/C/D/E/K/P/S) `agent.py`'s `search_catalog` tool already
  documents to the model; both the codes and original labels are kept.
- Records with `status != "ok"` (failed scrapes at the source) are dropped
  by default.

One known risk, carried over from the original plan and **not yet
resolved**: the feed has no field distinguishing "Individual Test
Solutions" from "Pre-packaged Job Solutions" (the `type=1` vs `type=2`
split on shl.com's own catalog page, which the assignment says to
exclude). Record names like `"Entry Level Cashier Solution"` strongly
suggest packaged solutions are present. `fetch_catalog.py` applies a
conservative, inspectable default filter — drop any record whose name
contains the word "Solution" — via `--keep-solutions` to disable it. This
is a naming heuristic, not a verified type field, and needs a real check
against the live site's `type=2` listing before `catalog.json` is treated
as fully scoped to Individual Test Solutions.

## 2. Retrieval: BM25, not embeddings

At ~380 short, keyword-dense rows ("Java", "SQL", "Accounts Payable"), BM25
over `name + description + test_type_labels + job_levels + languages`
matches this vocabulary at least as well as an embedding index, needs no
external API/vector DB, and is deterministic and inspectable — which matters
for a service graded on Recall@10 and hallucination rate, where "why did it
retrieve this" needs to be answerable. `retrieval.py` also supports hard
post-filters (max duration, test type, remote-only) layered on top of the
relevance ranking, so a constraint like "under 20 minutes" or "remote only"
is applied exactly, not left to the model's judgment.

## 3. Agent: tool-calling loop with a hard grounding guarantee

The model is given exactly two tools:

- `search_catalog(query, max_duration_minutes?, test_types?, remote_only?)`
  — queries the real index, can be called multiple times per turn.
- `final_response(reply, recommended_urls, end_of_conversation)` — the only
  way the model ends its turn.

The critical design choice: **the server, not the prompt, enforces
grounding**. Every URL the model recommends is checked against the set of
URLs actually returned by `search_catalog` calls made *in that turn*
(`_validate_recommendations`); anything else is silently dropped and logged.
This means hallucinated or memorized-from-training assessments cannot reach
the user even if the model ignores its system prompt — the failure mode
becomes "under-recommends," which is safe, not "recommends something fake,"
which isn't.

The system prompt handles the behavioral requirements that can't be
mechanically enforced the same way: ask one clarifying question on a vague
first turn instead of recommending; re-search and replace (not append to)
recommendations when the user changes a constraint; ground comparisons in
retrieved fields only; and refuse off-topic requests, legal-advice
questions, and prompt-injection attempts (including injections smuggled
inside pasted job descriptions), with an empty recommendation list on
refusal.

## 4. API contract & operational constraints

`POST /chat` is stateless — the client resends the full message history
every call, the server holds no session state, so the service is trivially
restart-safe and horizontally scalable. Turn cap: on the 9th user message
the server returns `end_of_conversation=true` *without calling the model at
all*; on the 8th (last allowed) it forces the model toward
`end_of_conversation=true` via an injected system-prompt note plus a
server-side override, so the cap holds even if the model forgets. A 30s
per-request timeout wraps the whole tool-calling loop via
`asyncio.wait_for`, returning a 504 with a clear message rather than hanging.
The tool loop itself is capped at `MAX_TOOL_ROUNDS` (4); if the model still
hasn't produced `final_response`, the last round strips `search_catalog`
from the tool list to force a decision.

## 5. Testing & what's verified vs. assumed

With `fastapi`, `pydantic`, `anthropic`, and `rank_bm25` installable in this
environment, most of the stack was exercised directly rather than by code
review alone:

- `fetch_catalog.py`'s transform logic, against a fixture covering the real
  feed's edge cases (blank/`"Variable"`/`"Untimed"`/`"0 minutes"` durations,
  multi-label `keys`, a `status != "ok"` record, remote/adaptive booleans) —
  duration parsing, test-type-code mapping, and status filtering all
  produced the expected `catalog.json` records.
- `retrieval.py`'s real BM25 search and hard-constraint filtering against
  that generated `catalog.json` — keyword search and `max_duration_minutes`
  filtering both returned the expected subsets.
- `main.py`'s FastAPI app via `TestClient` — `/health` returns 200; `/chat`
  correctly 422s on an empty message list and on a conversation whose last
  message isn't from the user (the `last_message_must_be_user` validator).
- `agent.py`'s full tool-calling loop, with the Anthropic client mocked to
  return scripted `search_catalog` / `final_response` tool calls — verified
  that a normal search-then-recommend turn returns the right
  `Recommendation`, and critically, that **a `final_response` recommending
  a URL never returned by `search_catalog` in that turn gets silently
  dropped by `_validate_recommendations`** rather than reaching the user —
  the core "no hallucination" guarantee actually holds in code, not just in
  the system prompt. Also verified the 9th-user-turn cap returns
  `end_of_conversation=true` without calling the model at all.

What's still assumed rather than verified: I don't have network access to
`tcp-us-prod-rnd.shl.com` from this sandbox, so `fetch_catalog.py` was
tested against a hand-built fixture mirroring the feed's schema (confirmed
via a one-off fetch of the live feed) rather than a live end-to-end run
against the full catalog; and no real Anthropic API call was made (only
the mocked client), so actual model behavior — whether it reliably asks
its one clarifying question, re-searches on constraint changes, and
refuses off-topic/legal/injection requests as instructed — is unverified
against the assignment's labeled conversation traces. Before treating this
as done: run `python fetch_catalog.py` for real, spot-check a sample of
`catalog.json` against the live SHL pages (especially the
Solutions-name-heuristic filter), and run the service with a real
`ANTHROPIC_API_KEY` against a handful of labeled traces to sanity-check
Recall@10 and the refusal/clarification probes.

## 6. Deployment

Build steps: `python fetch_catalog.py` (writes `catalog.json` from the live
feed) then `docker build .`, which bakes that `catalog.json` into a
single-container service (`uvicorn` + catalog). Any container host
(Fly.io, Render, Cloud Run, etc.) works for the "deploy publicly"
requirement — set `ANTHROPIC_API_KEY` and, if the catalog is refreshed
independently of the image, `SHL_CATALOG_PATH`.
