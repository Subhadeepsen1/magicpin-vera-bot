# composer.py - builds LLM prompts from the 4 contexts and calls the API

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
from typing import Any, Optional

# --- llm client ---

import time
import urllib.request
import urllib.error
import json
import os
from dataclasses import dataclass

@dataclass
class APIConfig:
    provider: str
    api_key: str
    model: str

def get_api_configs() -> list[APIConfig]:
    """Read API_1 through API_10 from env vars."""
    configs = []
    for i in range(1, 11):
        provider = os.environ.get(f"API_{i}_PROVIDER", "").strip().lower()
        api_key = os.environ.get(f"API_{i}_KEY", "").strip()
        model = os.environ.get(f"API_{i}_MODEL", "").strip()
        
        if provider and api_key:
            configs.append(APIConfig(provider=provider, api_key=api_key, model=model))
            
    # Fallback to old format if 10-tier isn't configured
    if not configs:
        if os.environ.get("GROQ_API_KEY"):
            configs.append(APIConfig("groq", os.environ.get("GROQ_API_KEY"), os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")))
        if os.environ.get("GEMINI_API_KEY"):
            configs.append(APIConfig("gemini", os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")))
            
    return configs

API_CONFIGS = get_api_configs()

def _llm_complete(prompt: str, system: str = "", max_tokens: int = 800, temperature: float = 0.3) -> str:
    """Try each API key in order, fall back on 429s."""
    if not API_CONFIGS:
        raise Exception("No valid API configurations found in environment variables.")

    errors = []
    for idx, config in enumerate(API_CONFIGS):
        try:
            if config.provider == "groq":
                return _groq_complete(prompt, system, max_tokens, temperature, config)
            elif config.provider == "gemini":
                return _gemini_complete(prompt, system, max_tokens, temperature, config)
            elif config.provider == "openai":
                return _openai_complete(prompt, system, max_tokens, temperature, config)
            elif config.provider == "deepseek":
                return _deepseek_complete(prompt, system, max_tokens, temperature, config)
            else:
                print(f"[Tier {idx+1}] Unknown provider: {config.provider}. Skipping.")
        except Exception as e:
            print(f"[Tier {idx+1}] {config.provider} failed: {e}. Cascading to next tier...")
            errors.append(str(e))
    
    raise Exception(f"All {len(API_CONFIGS)} fallback tiers exhausted. Errors: {'; '.join(errors)}")

def _groq_complete(prompt: str, system: str, max_tokens: int, temperature: float, config: APIConfig) -> str:
    """Groq API call (uses urllib instead of requests to dodge cloudflare)."""
    model = config.model or "llama-3.3-70b-versatile"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions", data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    )

    for attempt in range(2): # Reduce attempts to 2 so it cascades faster
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 1:
                time.sleep(2)
                continue
            raise Exception(f"Groq Error {e.code}: {e.read().decode()[:200]}")

def _gemini_complete(prompt: str, system: str, max_tokens: int, temperature: float, config: APIConfig) -> str:
    """Google Gemini API call."""
    model = config.model or "gemini-2.0-flash"
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    body = json.dumps({
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }).encode("utf-8")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.api_key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    for attempt in range(2):
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 1:
                time.sleep(2)
                continue
            raise Exception(f"Gemini Error {e.code}: {e.read().decode()[:200]}")

def _openai_complete(prompt: str, system: str, max_tokens: int, temperature: float, config: APIConfig) -> str:
    """OpenAI API call."""
    model = config.model or "gpt-4o-mini"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise Exception(f"OpenAI Error {e.code}: {e.read().decode()[:200]}")

def _deepseek_complete(prompt: str, system: str, max_tokens: int, temperature: float, config: APIConfig) -> str:
    """DeepSeek API call."""
    model = config.model or "deepseek-chat"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.deepseek.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise Exception(f"DeepSeek Error {e.code}: {e.read().decode()[:200]}")


# --- prompt builders ---

def _build_category_summary(category: dict) -> str:
    """Format category data for the LLM prompt."""
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest_items = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])
    offers = category.get("offer_catalog", [])

    digest_summary = "\n".join(
        f"  - [{d.get('kind','?')}] {d.get('title','')} (Source: {d.get('source','?')})"
        + (f" | Trial N={d['trial_n']}" if 'trial_n' in d else "")
        + (f" | Summary: {d.get('summary','')}" if d.get('summary') else "")
        for d in digest_items[:5]
    )

    return f"""CATEGORY: {category.get('slug', '?')}
Voice tone: {voice.get('tone', '?')}
Register: {voice.get('register', '?')}
Allowed vocabulary: {', '.join(voice.get('vocab_allowed', [])[:10])}
Taboo words/phrases: {', '.join(voice.get('vocab_taboo', [])[:6])}
Peer stats: avg_rating={peer.get('avg_rating','?')}, avg_ctr={peer.get('avg_ctr','?')}, avg_reviews={peer.get('avg_review_count','?')}, avg_views_30d={peer.get('avg_views_30d','?')}
Offer catalog templates: {', '.join(o.get('title','') for o in offers[:5])}
Seasonal beats: {'; '.join(s.get('month_range','') + ': ' + s.get('note','') for s in seasonal[:3])}
Trend signals: {'; '.join(t.get('query','') + ' ' + str(t.get('delta_yoy','')) + ' YoY' for t in trends[:3])}
Weekly digest items:
{digest_summary}"""


def _build_merchant_summary(merchant: dict) -> str:
    """Format merchant data for the LLM prompt."""
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    sub = merchant.get("subscription", {})
    offers = merchant.get("offers", [])
    cust_agg = merchant.get("customer_aggregate", {})
    signals = merchant.get("signals", [])
    conv_hist = merchant.get("conversation_history", [])
    reviews = merchant.get("review_themes", [])

    active_offers = [o for o in offers if o.get("status") == "active"]
    expired_offers = [o for o in offers if o.get("status") == "expired"]

    conv_summary = ""
    if conv_hist:
        conv_summary = "\nRecent conversation with Vera:\n" + "\n".join(
            f"  [{c.get('from','?')}] {c.get('body','')[:120]}... (engagement: {c.get('engagement','?')})"
            for c in conv_hist[-3:]
        )

    review_summary = ""
    if reviews:
        review_summary = "\nReview themes (30d): " + "; ".join(
            f"{r.get('theme','?')} ({r.get('sentiment','?')}, {r.get('occurrences_30d','?')}x)"
            + (f" — \"{r.get('common_quote','')}\"" if r.get('common_quote') else "")
            for r in reviews[:3]
        )

    return f"""MERCHANT: {identity.get('name', '?')}
Owner: {identity.get('owner_first_name', '?')}
City: {identity.get('city', '?')}, Locality: {identity.get('locality', '?')}
Languages: {', '.join(identity.get('languages', ['en']))}
Verified: {identity.get('verified', '?')}
Subscription: {sub.get('status','?')} ({sub.get('plan','?')}), {sub.get('days_remaining','?')} days remaining
Performance (30d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, directions={perf.get('directions','?')}, CTR={perf.get('ctr','?')}, leads={perf.get('leads','?')}
7d deltas: views {delta.get('views_pct','?')}, calls {delta.get('calls_pct','?')}
Active offers: {', '.join(o.get('title','') for o in active_offers) or 'NONE'}
Expired offers: {', '.join(o.get('title','') for o in expired_offers) or 'none'}
Customer aggregate: {json.dumps(cust_agg)}
Signals: {', '.join(str(s) for s in signals) or 'none'}
{review_summary}
{conv_summary}"""


def _build_trigger_summary(trigger: dict, category: dict) -> str:
    """Format trigger payload, resolving any digest item refs."""
    payload = trigger.get("payload", {})

    # Resolve digest item references
    top_item_id = payload.get("top_item_id")
    digest_detail = ""
    if top_item_id and category:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_detail = f"""
Referenced digest item:
  Title: {d.get('title','')}
  Source: {d.get('source','')}
  Summary: {d.get('summary','')}
  Trial N: {d.get('trial_n', 'N/A')}
  Patient segment: {d.get('patient_segment', 'N/A')}
  Actionable: {d.get('actionable','')}"""
                break

    return f"""TRIGGER:
ID: {trigger.get('id', '?')}
Kind: {trigger.get('kind', '?')}
Scope: {trigger.get('scope', '?')}
Source: {trigger.get('source', '?')}
Urgency: {trigger.get('urgency', '?')}/5
Suppression key: {trigger.get('suppression_key', '?')}
Payload: {json.dumps(payload, indent=2, default=str)}
{digest_detail}"""


def _build_customer_summary(customer: dict) -> str:
    """Format customer data (if present) for customer-facing messages."""
    if not customer:
        return "CUSTOMER: None (this is a merchant-facing message)"

    identity = customer.get("identity", {})
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})
    consent = customer.get("consent", {})

    return f"""CUSTOMER (this message is customer-facing, sent on behalf of the merchant):
Name: {identity.get('name', '?')}
Language preference: {identity.get('language_pref', 'en')}
Age band: {identity.get('age_band', '?')}
Senior citizen: {identity.get('senior_citizen', False)}
State: {customer.get('state', '?')}
Relationship: first_visit={rel.get('first_visit','?')}, last_visit={rel.get('last_visit','?')}, visits_total={rel.get('visits_total','?')}
Services received: {', '.join(str(s) for s in rel.get('services_received', [])[:6])}
Lifetime value: ₹{rel.get('lifetime_value', '?')}
Preferred slots: {prefs.get('preferred_slots', '?')}
Channel: {prefs.get('channel', 'whatsapp')}
Consent scope: {', '.join(consent.get('scope', []))}
Extra preferences: {json.dumps({k:v for k,v in prefs.items() if k not in ('channel','reminder_opt_in','preferred_slots')}, default=str) if prefs else 'none'}"""


# --- system prompt ---

SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant assistant. You compose WhatsApp messages for Indian merchants and their customers.

YOUR CORE RULES:
1. SPECIFICITY: Always anchor on verifiable facts from the context — numbers, dates, percentages, source citations, prices. "Dental Cleaning @ ₹299" beats "discount on dental services". "2,100-patient trial" beats "research shows".
2. CATEGORY FIT: Match voice exactly to the category. Dentists = peer-clinical, no overclaim. Salons = warm-practical. Restaurants = operator-to-operator. Gyms = coaching. Pharmacies = trustworthy-precise.
3. MERCHANT FIT: Personalize to THIS merchant. Use their owner's first name, their actual data (numbers, offers, signals), their language preference. Never fabricate data not in the context.
4. TRIGGER RELEVANCE: State WHY you're messaging NOW. The trigger is the reason — reference it clearly. Not generic "improve your profile".
5. ENGAGEMENT COMPULSION: Use one or more levers — curiosity, loss aversion, social proof, effort externalization, asking-the-merchant, single binary CTA.
6. HINDI-ENGLISH MIX: If the merchant's languages include "hi", naturally code-mix Hindi and English. Don't force it — be natural.
7. NO FABRICATION: If data isn't in the contexts, don't invent it. No fake offers, research citations, competitor names, or statistics.
8. CONCISE: No long preambles ("I hope you're doing well..."). Get to the point. Lead with the hook.
9. SINGLE CTA: One clear call-to-action at the end. Binary (YES/STOP) for action triggers. Open-ended for information triggers. None acceptable for pure-info.
10. NO TABOO WORDS: Never use words from the category's taboo list.

ANTI-PATTERNS TO AVOID:
- "Flat X% off" when service+price is available
- Multiple CTAs in one message
- Promotional tone for clinical categories
- Re-introducing yourself after the first message
- Long messages when short would work

OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON object (no markdown, no code fences):
{
  "body": "the WhatsApp message body",
  "cta": "binary_yes_no" | "open_ended" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "from the trigger",
  "rationale": "1-2 sentence explanation of why this message, what compulsion levers used"
}"""


# --- compose ---

def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    """
    Compose a message using the 4-context framework.
    Returns dict with keys: body, cta, send_as, suppression_key, rationale.
    """
    is_customer_facing = trigger.get("scope") == "customer" and customer is not None

    user_prompt = f"""Compose a WhatsApp message using these contexts:

{_build_category_summary(category)}

{_build_merchant_summary(merchant)}

{_build_trigger_summary(trigger, category)}

{_build_customer_summary(customer)}

IMPORTANT NOTES:
- send_as should be "{'merchant_on_behalf' if is_customer_facing else 'vera'}"
- {'This is CUSTOMER-FACING: message is sent FROM the merchant WA number TO the customer. Use the merchant name as sender, warm tone, honor customer language pref.' if is_customer_facing else 'This is MERCHANT-FACING: message is sent FROM Vera TO the merchant.'}
- suppression_key should be: "{trigger.get('suppression_key', '')}"
- Keep body concise but specific. Lead with the hook, end with the CTA.

Now compose the message. Respond with ONLY a JSON object."""

    try:
        raw = _llm_complete(user_prompt, SYSTEM_PROMPT, max_tokens=800)
        return _parse_compose_response(raw, trigger)

    except Exception as e:
        # Fallback: return a basic message
        merchant_name = merchant.get("identity", {}).get("owner_first_name", "there")
        return {
            "body": f"Hi {merchant_name}, checking in with an update for your business. Reply YES if you'd like details.",
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if is_customer_facing else "vera",
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": f"Fallback due to LLM error: {str(e)[:100]}",
        }


def _parse_compose_response(raw: str, trigger: dict) -> dict:
    """Try to parse JSON from the LLM output, fall back to raw text."""
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

    # Try to find JSON object
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            data = json.loads(match.group())
            # Validate and normalize
            return {
                "body": data.get("body", ""),
                "cta": data.get("cta", "open_ended"),
                "send_as": data.get("send_as", "vera"),
                "suppression_key": data.get("suppression_key", trigger.get("suppression_key", "")),
                "rationale": data.get("rationale", ""),
            }
        except json.JSONDecodeError:
            pass

    # If parsing fails, use raw as body
    return {
        "body": raw[:500],
        "cta": "open_ended",
        "send_as": "vera",
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": "LLM response was not valid JSON; used raw text",
    }


# --- reply handling ---

REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant, handling a multi-turn WhatsApp conversation.

CRITICAL BEHAVIORS:
1. AUTO-REPLY DETECTION: If the message looks like a WhatsApp Business canned auto-reply ("Thank you for contacting...", "Our team will respond shortly", etc.), detect it and respond appropriately.
   - First auto-reply: Send one short message acknowledging it, nudge the owner.
   - Second auto-reply: Wait (back off).
   - Third+ auto-reply: End the conversation.

2. INTENT TRANSITION: If the merchant explicitly commits ("ok let's do it", "yes", "go ahead", "what's next"), IMMEDIATELY switch to action mode. Do NOT ask another qualifying question. Start executing.

3. HOSTILE/OPT-OUT: If the merchant says "stop", "not interested", "don't message me", end gracefully with a brief apology.

4. OFF-TOPIC: If the merchant asks something outside your scope (GST, legal, personal), politely decline and redirect to the original thread.

5. ENGAGED REPLY: If the merchant is genuinely engaged, continue the conversation naturally. Add value, don't repeat yourself.

RESPOND WITH ONLY a valid JSON object:
{
  "action": "send" | "wait" | "end",
  "body": "reply text (only if action=send)",
  "cta": "binary_yes_no" | "open_ended" | "none",
  "wait_seconds": 1800 (only if action=wait),
  "rationale": "1-2 sentence explanation"
}"""


def compose_reply(conversation_history: list[dict], merchant: dict,
                  category: dict, trigger: dict,
                  merchant_message: str, turn_number: int,
                  auto_reply_count: int = 0,
                  customer: Optional[dict] = None) -> dict:
    """
    Compose a reply to a merchant/customer message in an ongoing conversation.
    """
    identity = merchant.get("identity", {})

    conv_text = "\n".join(
        f"[{t.get('from', '?').upper()}] {t.get('body', '')}"
        for t in conversation_history[-6:]  # last 6 turns for context
    )

    user_prompt = f"""CONVERSATION CONTEXT:
Merchant: {identity.get('name', '?')} (Owner: {identity.get('owner_first_name', '?')})
Category: {category.get('slug', '?')} (Voice: {category.get('voice', {}).get('tone', '?')})
Original trigger: {trigger.get('kind', '?')}
Turn number: {turn_number}
Auto-reply detections so far in this conversation: {auto_reply_count}

CONVERSATION SO FAR:
{conv_text}

LATEST MESSAGE FROM MERCHANT:
"{merchant_message}"

Respond appropriately. Remember:
- If auto-reply detected {auto_reply_count} times already, be more aggressive about waiting/ending.
- If merchant explicitly commits, switch to ACTION mode immediately.
- Never repeat the same body text you've already sent.
- Keep responses concise.

Respond with ONLY a JSON object."""

    try:
        raw = _llm_complete(user_prompt, REPLY_SYSTEM, max_tokens=600)
        return _parse_reply_response(raw)

    except Exception as e:
        return {
            "action": "send",
            "body": "Got it, let me follow up on that.",
            "cta": "open_ended",
            "rationale": f"Fallback due to LLM error: {str(e)[:100]}",
        }


def _parse_reply_response(raw: str) -> dict:
    """Parse JSON from the reply LLM output."""
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            data = json.loads(match.group())
            action = data.get("action", "send")
            result = {
                "action": action,
                "rationale": data.get("rationale", ""),
            }
            if action == "send":
                result["body"] = data.get("body", "")
                result["cta"] = data.get("cta", "open_ended")
            elif action == "wait":
                result["wait_seconds"] = data.get("wait_seconds", 1800)
            return result
        except json.JSONDecodeError:
            pass

    return {
        "action": "send",
        "body": raw[:300],
        "cta": "open_ended",
        "rationale": "LLM response was not valid JSON",
    }


# --- auto-reply / hostile / intent detection ---

AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "thanks for reaching out",
    "our team will respond",
    "we will get back to you",
    "your message has been received",
    "automated response",
    "auto-reply",
    "our working hours",
    "we are currently unavailable",
    "thank you for your message",
    "we have received your message",
    "हमने आपका संदेश प्राप्त कर लिया",
    "हमारी टीम जल्द ही आपसे संपर्क करेगी",
    "aapki jaankari ke liye",
    "shukriya",
    "main ek automated assistant hoon",
]


def is_auto_reply(message: str) -> bool:
    """Check if the message looks like a WA Business canned auto-reply."""
    msg_lower = message.lower().strip()
    return any(pattern in msg_lower for pattern in AUTO_REPLY_PATTERNS)


def is_hostile(message: str) -> bool:
    """Check if merchant wants us to stop messaging."""
    msg_lower = message.lower().strip()
    hostile_patterns = [
        "stop messaging", "don't message", "not interested",
        "stop sending", "unsubscribe", "block", "spam",
        "leave me alone", "stop bothering", "useless",
    ]
    return any(p in msg_lower for p in hostile_patterns)


def is_intent_commit(message: str) -> bool:
    """Check if merchant said yes / wants to proceed."""
    msg_lower = message.lower().strip()
    commit_patterns = [
        "yes", "ok let", "let's do it", "go ahead", "proceed",
        "what's next", "whats next", "do it", "confirm",
        "haan", "kar do", "chalo", "theek hai", "ok done",
    ]
    return any(p in msg_lower for p in commit_patterns)
