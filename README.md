# Vera Bot — magicpin AI Challenge

## What it does

Runs a FastAPI server that acts as "Vera", magicpin's merchant assistant. Takes in merchant/category/trigger/customer context via HTTP, composes WhatsApp messages using an LLM, and handles multi-turn conversations.

Built around the 4-context framework from the challenge brief. Uses multiple API keys with automatic failover so it doesn't die when one key gets rate-limited.

### Folder layout

```
├── app/
│   ├── bot.py              # FastAPI routes (healthz, context, tick, reply)
│   ├── composer.py          # LLM calls + prompt construction
│   └── context_store.py     # in-memory state (contexts, convos, suppression)
│
├── scripts/
│   ├── generate_submission.py   # offline batch run -> submission.jsonl
│   └── judge_simulator.py       # local test harness
│
├── data/
│   ├── dataset/             # seed data + expanded test fixtures
│   └── submission.jsonl     # output from generate_submission
│
├── docs/                    # challenge briefs, research notes, examples
├── .env                     # API keys (not committed to public repos)
└── requirements.txt
```

### How it works

1. Judge pushes context (category voice rules, merchant data, triggers) via `/v1/context`
2. Judge calls `/v1/tick` with available triggers → bot composes proactive messages
3. Merchant replies come in via `/v1/reply` → bot handles multi-turn (auto-reply detection, hostile opt-out, intent commits)

The composer tries Groq first, then falls back through up to 9 more API keys (Gemini, OpenAI, DeepSeek) if it hits rate limits. Config lives in `.env` as `API_1_*` through `API_10_*`.

### Design tradeoffs

- **In-memory state** — no DB, no persistence across restarts. Good enough for the 60-min test window.
- **Single prompt for all trigger types** — a prod system would have per-trigger prompt variants. We rely on the system prompt + rich context instead.
- **Heuristics before LLM** — auto-reply and hostile detection happen with pattern matching, not LLM calls. Faster and more predictable.

### Running locally

```bash
uvicorn app.bot:app --host 0.0.0.0 --port 8080

# test with the judge
python scripts/judge_simulator.py

# batch generate submission file
python scripts/generate_submission.py
```
