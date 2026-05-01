# context_store.py - in-memory state for contexts, conversations, suppression

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional


class ContextStore:
    """Stores all 4 context scopes + conversations in memory."""

    def __init__(self):
        # (scope, context_id) -> {version, payload, stored_at}
        self.contexts: dict[tuple[str, str], dict[str, Any]] = {}

        # conversation_id -> list of turns [{from, body, ts, ...}]
        self.conversations: dict[str, list[dict]] = {}

        # conversation_id -> metadata {merchant_id, customer_id, trigger_id, ended, ...}
        self.conversation_meta: dict[str, dict] = {}

        # suppression_key -> timestamp of last send (prevent duplicate sends)
        self.suppressed: dict[str, float] = {}

        # Track which triggers have been acted on
        self.acted_triggers: set[str] = set()

        self.start_time = time.time()

    # --- context CRUD ---

    def push_context(self, scope: str, context_id: str, version: int,
                     payload: dict) -> tuple[bool, Optional[int]]:
        """
        Store or update a context.
        Returns (accepted, current_version_if_rejected).
        """
        key = (scope, context_id)
        existing = self.contexts.get(key)

        if existing and existing["version"] >= version:
            return False, existing["version"]

        self.contexts[key] = {
            "version": version,
            "payload": payload,
            "stored_at": datetime.utcnow().isoformat() + "Z",
        }
        return True, None

    def get_context(self, scope: str, context_id: str) -> Optional[dict]:
        """Get the payload for a context, or None."""
        entry = self.contexts.get((scope, context_id))
        return entry["payload"] if entry else None

    def get_context_counts(self) -> dict[str, int]:
        """Count contexts by scope."""
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        for (scope, _) in self.contexts:
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    # --- convenience lookups ---──

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        return self.get_context("trigger", trigger_id)

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        return self.get_context("merchant", merchant_id)

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self.get_context("customer", customer_id)

    def get_category(self, slug: str) -> Optional[dict]:
        return self.get_context("category", slug)

    def get_category_for_merchant(self, merchant: dict) -> Optional[dict]:
        slug = merchant.get("category_slug", "")
        return self.get_category(slug)

    # --- conversation tracking ---──

    def start_conversation(self, conv_id: str, merchant_id: str,
                           customer_id: Optional[str], trigger_id: str):
        """Register a new conversation."""
        self.conversation_meta[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trigger_id,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "ended": False,
            "auto_reply_count": 0,
        }
        self.conversations[conv_id] = []

    def add_turn(self, conv_id: str, from_role: str, body: str,
                 turn_number: int = 0):
        """Add a turn to a conversation."""
        self.conversations.setdefault(conv_id, []).append({
            "from": from_role,
            "body": body,
            "ts": datetime.utcnow().isoformat() + "Z",
            "turn_number": turn_number,
        })

    def get_conversation(self, conv_id: str) -> list[dict]:
        """Get all turns for a conversation."""
        return self.conversations.get(conv_id, [])

    def get_conversation_meta(self, conv_id: str) -> Optional[dict]:
        return self.conversation_meta.get(conv_id)

    def end_conversation(self, conv_id: str):
        """Mark a conversation as ended."""
        if conv_id in self.conversation_meta:
            self.conversation_meta[conv_id]["ended"] = True

    def is_conversation_ended(self, conv_id: str) -> bool:
        meta = self.conversation_meta.get(conv_id)
        return meta["ended"] if meta else False

    def increment_auto_reply(self, conv_id: str) -> int:
        """Increment auto-reply counter, return new count."""
        meta = self.conversation_meta.get(conv_id, {})
        count = meta.get("auto_reply_count", 0) + 1
        if conv_id in self.conversation_meta:
            self.conversation_meta[conv_id]["auto_reply_count"] = count
        return count

    # --- suppression ---──

    def is_suppressed(self, suppression_key: str) -> bool:
        return suppression_key in self.suppressed

    def suppress(self, suppression_key: str):
        self.suppressed[suppression_key] = time.time()

    def mark_trigger_acted(self, trigger_id: str):
        self.acted_triggers.add(trigger_id)

    def is_trigger_acted(self, trigger_id: str) -> bool:
        return trigger_id in self.acted_triggers

    # --- misc ---

    def uptime_seconds(self) -> int:
        return int(time.time() - self.start_time)

    def get_last_bot_message(self, conv_id: str) -> Optional[str]:
        """Get the last message the bot sent in a conversation."""
        turns = self.conversations.get(conv_id, [])
        for turn in reversed(turns):
            if turn["from"] in ("vera", "bot", "merchant_on_behalf"):
                return turn["body"]
        return None
