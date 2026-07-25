"""Family bot plugin for zoozl server.

A conversational family assistant built on the OpenAI Agents SDK. A single
``Agent`` owns the whole conversation; per-user history is persisted in a local
SQLite session store and replayed to the model through a bounded sliding window
so the prompt never grows without limit.

Usage:
    Start with: python -m zoozl --conf family-bot.toml
"""

import logging
import os

from agents import Agent, Runner, SQLiteSession, set_default_openai_key
from zoozl.chatbot import Interface


log = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful family assistant. "
    "You help organise family matters such as schedules, tasks, "
    "shopping lists, and family communications."
)

GREETING = (
    "Hello! I'm your family assistant. I can help with schedules, tasks, "
    "shopping lists, and keeping the family organised. What do you need?"
)


class WindowedSession(SQLiteSession):
    """SQLiteSession that only replays the last N items to the model.

    The full conversation is still persisted, but each turn sends just a
    sliding window so the prompt does not grow without bound. The window is
    trimmed forward until it starts on a user message, so it never begins in
    the middle of an assistant/tool-call sequence (which the model rejects).
    """

    def __init__(self, session_id, db_path, window_size=10):
        """Store the window size alongside the usual session parameters."""
        super().__init__(session_id=session_id, db_path=db_path)
        self.window_size = window_size

    async def get_items(self, limit: int | None = None):
        """Return at most ``window_size`` recent items, starting on a user turn."""
        if limit is None:
            limit = self.window_size
        items = await super().get_items(limit=limit)
        while items and items[0].get("role") != "user":
            items.pop(0)
        return items


class FamilyAssistant(Interface):
    """Family assistant powered by the OpenAI Agents SDK.

    A single agent handles every message. Conversation history is persisted per
    talker (the WhatsApp ``wa_id``) in a SQLite session store and replayed to
    the model through a bounded sliding window.
    """

    # Owning greet/help makes every message (and the initial greeting) route to
    # this single agent under the current zoozl router. ``cancel`` is
    # deliberately NOT claimed: zoozl invokes the cancel handler as a pre-check
    # on every turn, so owning it would run the agent twice per message and
    # send a duplicate reply.
    aliases = {"greet", "help"}

    def load(self, root):
        """Configure the OpenAI key, build the agent, locate the session store.

        Configuration is read from the ``[family_bot]`` section of the zoozl
        TOML config file.  ``openai_api_key`` is optional: when omitted the
        Agents SDK falls back to the ``OPENAI_API_KEY`` environment variable.
        """
        conf = root.conf.get("family_bot", {})
        api_key = conf.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            set_default_openai_key(api_key)
        self._model = conf.get("model", "gpt-4o-mini")
        self._system_prompt = conf.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        self._session_db = conf.get("session_database", "family_bot_sessions.db")
        self._history_window = conf.get("history_window", 10)
        self._agent = Agent(
            name="Family assistant",
            instructions=self._system_prompt,
            model=self._model,
        )

    async def consume(self, package):
        """Hand the message to the agent and reply on the same channel.

        History is persisted and windowed per talker by ``WindowedSession``, so
        context is preserved across turns while the prompt stays bounded.
        """
        text = package.last_message_text
        if not text:
            # Initial greeting on connect (no user message yet).
            package.callback(GREETING)
            return
        session = WindowedSession(
            package.talker, self._session_db, self._history_window
        )
        try:
            result = await Runner.run(
                self._agent, text, session=session, context=package
            )
            reply = result.final_output
        except Exception as exc:  # noqa: BLE001 - keep the webhook resilient
            log.error("Agent run failed: %s", exc)
            reply = "Sorry, I am unable to respond right now. Please try again later."
        package.callback(reply)
