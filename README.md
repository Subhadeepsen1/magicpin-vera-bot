# Vera — magicpin AI Merchant Assistant

Vera is a production-grade AI assistant designed to help magicpin merchants engage with their customers and manage their business via WhatsApp.

## 🚀 Public Bot URL
**URL**: [https://magicpin-vera-bot-tbbl.onrender.com](https://magicpin-vera-bot-tbbl.onrender.com)

---

## 🧠 Technical Approach

### 1. 10-Tier Cascading Fallback Engine
To ensure near-100% availability during high-traffic events or API outages, Vera uses a **Dynamic Orchestration Layer**. It rotates through up to 10 distinct API configurations (Groq, Gemini, OpenAI, DeepSeek) defined in environment variables. If a primary provider returns a `429 Rate Limit` or `500 Error`, the system cascades to the next tier in milliseconds.

### 2. Heuristic-First Intelligence
Before invoking an LLM, Vera runs a lightweight heuristic scan to:
- **Detect Auto-replies**: Prevents "infinite loops" between bots.
- **Detect Opt-outs**: Automatically handles "Stop" or "Not interested" messages to respect merchant privacy.
- **Track Intent Commitment**: Identifies when a merchant has already said "Yes" to minimize redundant sales pitches.

### 3. Contextual Persona Mapping
The system uses a 4-context framework (**Category**, **Merchant**, **Trigger**, and **Customer**) to generate hyper-personalized responses. The prompt engineering is tuned for a professional yet friendly "Hinglish" (Hindi-English) tone, reflecting the real-world communication style of Indian merchants.

---

## 🤖 Model Choice

- **Primary Model**: `Llama-3.3-70b-versatile` (via Groq)
  - **Reason**: Exceptional speed (low latency is critical for chat) and strong performance in code-mixed (Hindi-English) languages.
- **Secondary Model**: `Gemini-2.0-Flash` (via Google AI Studio)
  - **Reason**: Massive context window and extremely high reliability as a fallback tier.

These models were selected because they provide the best balance of **linguistic nuance** (understanding Indian cultural context) and **inference speed**.

---

## ⚖️ Tradeoffs

- **Consistency vs. Availability**: We chose a cascading fallback approach. While switching models might lead to slight variations in tone, we prioritized **Availability** (always responding to the merchant) over strict model consistency.
- **Latency vs. Cost**: We implemented local heuristics for common intents. This adds slight complexity to the code but significantly reduces **Latency** and **API Costs** by skipping the LLM for simple "Yes/No" or "Stop" responses.
- **State Management**: We used an in-memory `ContextStore` for this challenge. This provides sub-millisecond state lookups but would need to be traded off for a distributed database (like Redis) in a multi-region production environment.

---

## 🛠️ Project Structure
- `app/`: Core FastAPI server, state management, and LLM composer.
- `scripts/`: Testing tools and submission generators.
- `data/`: The context datasets (Categories, Merchants, Triggers).
- `docs/`: Original research and challenge briefs.
