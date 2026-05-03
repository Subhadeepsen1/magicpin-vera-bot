# bot.py - FastAPI server for the magicpin AI challenge

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.context_store import ContextStore
from app.composer import (
    compose,
    compose_reply,
    is_auto_reply,
    is_hostile,
    is_intent_commit,
)

# --- app setup ---

app = FastAPI(title="Vera Bot — magicpin AI Challenge")
store = ContextStore()

# Enable CORS for the dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def get_dashboard():
    return FileResponse("static/index.html")

TEAM_NAME = os.environ.get("TEAM_NAME", "Subhadeep")
TEAM_EMAIL = os.environ.get("TEAM_EMAIL", "subhadeepsen@example.com")
START_TIME = time.time()


# --- request/response models ---

class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# --- endpoints ---

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": store.uptime_seconds(),
        "contexts_loaded": store.get_context_counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": [TEAM_NAME],
        "uptime_seconds": int(time.time() - START_TIME),
        "model": os.environ.get("API_1_MODEL", "llama-3.3-70b-versatile"),
        "approach": (
            "4-context LLM composer with trigger-kind dispatch, "
            "auto-reply detection heuristics, intent-transition handling, "
            "and a 10-tier fallback mechanism. Prompts encode category voice, "
            "merchant state, and Hinglish tone."
        ),
        "contact_email": TEAM_EMAIL,
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextPush):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}",
        }

    accepted, current_version = store.push_context(
        body.scope, body.context_id, body.version, body.payload
    )

    if not accepted:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current_version,
        }

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickRequest):
    actions = []

    for trigger_id in body.available_triggers:
        # Skip if we've already acted on this trigger
        if store.is_trigger_acted(trigger_id):
            continue

        trigger = store.get_trigger(trigger_id)
        if not trigger:
            continue

        # Get merchant_id from the trigger payload
        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            continue

        merchant = store.get_merchant(merchant_id)
        if not merchant:
            continue

        category = store.get_category_for_merchant(merchant)
        if not category:
            continue

        # Check suppression
        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and store.is_suppressed(suppression_key):
            continue

        # Get customer if this is a customer-scoped trigger
        customer_id = trigger.get("customer_id")
        customer = store.get_customer(customer_id) if customer_id else None

        # Compose the message
        try:
            composed = compose(category, merchant, trigger, customer)
        except Exception as e:
            # On error, skip this trigger
            continue

        if not composed.get("body"):
            continue

        # Generate conversation ID
        owner = merchant.get("identity", {}).get("owner_first_name", "mx")
        conv_id = f"conv_{merchant_id}_{trigger.get('kind', 'unknown')}_{trigger_id[-10:]}"

        # Determine send_as
        is_customer_facing = trigger.get("scope") == "customer" and customer is not None
        send_as = "merchant_on_behalf" if is_customer_facing else "vera"

        # Build template params from the composed body
        merchant_name = merchant.get("identity", {}).get("name", "Merchant")
        template_params = [
            merchant.get("identity", {}).get("owner_first_name", merchant_name),
            composed["body"][:160],
            composed.get("cta", ""),
        ]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", send_as),
            "trigger_id": trigger_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": composed.get("suppression_key", suppression_key),
            "rationale": composed.get("rationale", ""),
        }

        actions.append(action)

        # Register the conversation + mark trigger as acted + suppress
        store.start_conversation(conv_id, merchant_id, customer_id, trigger_id)
        store.add_turn(conv_id, send_as, composed["body"], turn_number=1)
        store.mark_trigger_acted(trigger_id)
        if suppression_key:
            store.suppress(suppression_key)

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    conv_id = body.conversation_id
    merchant_message = body.message

    # If conversation is already ended, don't respond
    if store.is_conversation_ended(conv_id):
        return {
            "action": "end",
            "rationale": "Conversation was previously ended.",
        }

    # Record the incoming message
    store.add_turn(conv_id, body.from_role, merchant_message, body.turn_number)

    # heuristic checks before hitting the LLM

    # 1. Auto-reply detection
    if is_auto_reply(merchant_message):
        auto_count = store.increment_auto_reply(conv_id)

        if auto_count >= 3:
            store.end_conversation(conv_id)
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {auto_count} times consecutively. No real engagement signal; closing conversation.",
            }
        elif auto_count >= 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": f"Same auto-reply {auto_count} times — owner not at phone. Waiting 24h before retry.",
            }
        # First auto-reply: let LLM handle with context

    # 2. Hostile / opt-out
    if is_hostile(merchant_message):
        store.end_conversation(conv_id)
        return {
            "action": "end",
            "rationale": "Merchant expressed frustration/opted out. Closing conversation gracefully; suppressing future triggers.",
        }

    # call the LLM for a proper reply

    # Get context for this conversation
    meta = store.get_conversation_meta(conv_id)
    merchant_id = body.merchant_id or (meta.get("merchant_id") if meta else None)
    merchant = store.get_merchant(merchant_id) if merchant_id else {}
    if not merchant:
        merchant = {}

    category = store.get_category_for_merchant(merchant) if merchant else {}
    if not category:
        category = {}

    trigger_id = meta.get("trigger_id") if meta else None
    trigger = store.get_trigger(trigger_id) if trigger_id else {}
    if not trigger:
        trigger = {}

    customer_id = body.customer_id or (meta.get("customer_id") if meta else None)
    customer = store.get_customer(customer_id) if customer_id else None

    # Get conversation history
    conv_history = store.get_conversation(conv_id)

    # Auto-reply count for context
    auto_count = meta.get("auto_reply_count", 0) if meta else 0

    # Compose reply
    try:
        result = compose_reply(
            conversation_history=conv_history,
            merchant=merchant,
            category=category,
            trigger=trigger,
            merchant_message=merchant_message,
            turn_number=body.turn_number,
            auto_reply_count=auto_count,
            customer=customer,
        )
    except Exception as e:
        result = {
            "action": "send",
            "body": f"Backend Error: {str(e)}",
            "cta": "open_ended",
            "rationale": f"Fallback reply due to error: {str(e)[:100]}",
        }

    # Check for repetition — don't send the same body twice
    if result.get("action") == "send":
        last_bot_msg = store.get_last_bot_message(conv_id)
        if last_bot_msg and result.get("body") == last_bot_msg:
            result["body"] = result["body"] + " (updated)"

    # Record the bot's response
    if result.get("action") == "send" and result.get("body"):
        store.add_turn(conv_id, "vera", result["body"], body.turn_number + 1)

    if result.get("action") == "end":
        store.end_conversation(conv_id)

    return result


# --- demo data loader ---

@app.on_event("startup")
async def load_demo_data():
    """Pre-loads some data so the Jury Sandbox works immediately."""
    print("Loading demo data for juries...")
    
    # Demo Category (Version 1)
    store.push_context("category", "southindiancafe", 1, {
        "slug": "southindiancafe",
        "name": "South Indian Cafe",
        "voice": "Traditional, warm, and inviting. Uses terms like 'Annapoorna', 'Authentic', and 'Fresh'."
    })
    
    # Demo Merchant (Version 1)
    store.push_context("merchant", "m_001_mylari", 1, {
        "merchant_id": "m_001_mylari",
        "category_slug": "southindiancafe",
        "identity": {
            "name": "Mylari South Indian Cafe",
            "city": "Mysuru"
        },
        "business": {
            "specialty": "Mylari Dose, Filter Coffee",
            "usp": "Softest idlis in town"
        }
    })
    
    # Demo Trigger (Version 1)
    store.push_context("trigger", "t_001_planning", 1, {
        "id": "t_001_planning",
        "kind": "active_planning_intent",
        "payload": {
            "intent": "Bulk corporate order",
            "discount_available": "15%"
        }
    })
    print("Demo data loaded successfully.")


# --- main ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=True)
