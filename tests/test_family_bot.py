"""Tests for the family_bot plugin (OpenAI Agents SDK)."""

import os
import sys
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on the path so family_bot can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import family_bot


class TestWindowedSession(IsolatedAsyncioTestCase):
    """Tests for the bounded sliding-window session."""

    async def test_trims_forward_to_first_user_turn(self):
        """Leading non-user items are dropped so the window starts on a user turn."""
        raw = [
            {"role": "assistant", "content": "earlier reply"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        with patch.object(family_bot.SQLiteSession, "__init__", return_value=None), \
             patch.object(
                 family_bot.SQLiteSession,
                 "get_items",
                 new=AsyncMock(return_value=list(raw)),
             ):
            session = family_bot.WindowedSession("wa-1", "db", window_size=5)
            items = await session.get_items()

        self.assertEqual([i["role"] for i in items], ["user", "assistant"])

    async def test_uses_window_size_as_default_limit(self):
        """With no explicit limit, the configured window size bounds the replay."""
        base_get = AsyncMock(return_value=[])
        with patch.object(family_bot.SQLiteSession, "__init__", return_value=None), \
             patch.object(family_bot.SQLiteSession, "get_items", new=base_get):
            session = family_bot.WindowedSession("wa-1", "db", window_size=7)
            await session.get_items()

        base_get.assert_awaited_once_with(limit=7)


class TestFamilyAssistantLoad(IsolatedAsyncioTestCase):
    """Tests for the FamilyAssistant.load method."""

    def _make_root(self, conf=None):
        """Return a minimal mock InterfaceRoot."""
        root = MagicMock()
        root.conf = conf or {}
        return root

    @patch("family_bot.Agent")
    @patch("family_bot.set_default_openai_key")
    def test_load_defaults(self, mock_set_key, mock_agent):
        """Plugin loads with defaults when no config section is present."""
        with patch.dict(os.environ, {}, clear=True):
            assistant = family_bot.FamilyAssistant()
            assistant.load(self._make_root())

        mock_set_key.assert_not_called()  # no key in conf or environment
        self.assertEqual(assistant._model, "gpt-4o-mini")
        self.assertIn("family assistant", assistant._system_prompt)
        self.assertEqual(assistant._session_db, "family_bot_sessions.db")
        self.assertEqual(assistant._history_window, 10)
        kwargs = mock_agent.call_args.kwargs
        self.assertEqual(kwargs["instructions"], assistant._system_prompt)
        self.assertEqual(kwargs["model"], "gpt-4o-mini")

    @patch("family_bot.Agent")
    @patch("family_bot.set_default_openai_key")
    def test_load_custom_config(self, mock_set_key, mock_agent):
        """Plugin picks up custom settings from the config section."""
        conf = {
            "database": "/tmp/custom.db",
            "family_bot": {
                "openai_api_key": "sk-test",
                "model": "gpt-4o",
                "system_prompt": "Custom prompt",
                "history_window": 4,
            }
        }
        assistant = family_bot.FamilyAssistant()
        assistant.load(self._make_root(conf))

        mock_set_key.assert_called_once_with("sk-test")
        self.assertEqual(assistant._model, "gpt-4o")
        self.assertEqual(assistant._system_prompt, "Custom prompt")
        self.assertEqual(assistant._session_db, "/tmp/custom.db")
        self.assertEqual(assistant._history_window, 4)

    @patch("family_bot.Agent")
    @patch("family_bot.set_default_openai_key")
    def test_load_falls_back_to_env_key(self, mock_set_key, mock_agent):
        """When no key is in config, the OPENAI_API_KEY environment var is used."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}, clear=True):
            assistant = family_bot.FamilyAssistant()
            assistant.load(self._make_root())

        mock_set_key.assert_called_once_with("sk-env")

    def test_aliases(self):
        """FamilyAssistant handles ordinary messages via greet/help.

        It must NOT claim the ``cancel`` alias: zoozl invokes the cancel
        handler as a pre-check on every turn, so claiming it would run the
        agent twice and send a duplicate reply per message.
        """
        self.assertIn("greet", family_bot.FamilyAssistant.aliases)
        self.assertIn("help", family_bot.FamilyAssistant.aliases)
        self.assertNotIn("cancel", family_bot.FamilyAssistant.aliases)


class TestFamilyAssistantConsume(IsolatedAsyncioTestCase):
    """Tests for the FamilyAssistant.consume method."""

    def _make_assistant(self):
        """Return a loaded FamilyAssistant with a stubbed agent."""
        assistant = family_bot.FamilyAssistant()
        assistant._agent = MagicMock()
        assistant._session_db = "family_bot_sessions.db"
        assistant._history_window = 10
        return assistant

    def _make_package(self, user_text="Hello"):
        """Return a minimal mock Package."""
        package = MagicMock()
        package.last_message_text = user_text
        package.talker = "wa-123"
        package.callback = MagicMock()
        return package

    async def test_sends_agent_reply_to_callback(self):
        """consume() passes the agent's final output to package.callback."""
        assistant = self._make_assistant()
        package = self._make_package("Hello")
        run_result = MagicMock(final_output="Agent reply")

        with patch("family_bot.WindowedSession") as mock_session, patch(
            "family_bot.Runner.run", new=AsyncMock(return_value=run_result)
        ) as mock_run:
            await assistant.consume(package)

        mock_session.assert_called_once_with("wa-123", "family_bot_sessions.db", 10)
        mock_run.assert_awaited_once()
        package.callback.assert_called_once_with("Agent reply")

    async def test_empty_message_sends_greeting_without_running_agent(self):
        """An empty message (initial connect) greets without invoking the agent."""
        assistant = self._make_assistant()
        package = self._make_package("")

        with patch("family_bot.Runner.run", new=AsyncMock()) as mock_run:
            await assistant.consume(package)

        mock_run.assert_not_called()
        package.callback.assert_called_once_with(family_bot.GREETING)

    async def test_agent_error_returns_fallback_message(self):
        """When the agent run fails a friendly fallback is returned."""
        assistant = self._make_assistant()
        package = self._make_package("Hello")

        with patch("family_bot.WindowedSession"), patch(
            "family_bot.Runner.run", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            await assistant.consume(package)

        text = package.callback.call_args.args[0]
        self.assertIn("unable to respond", text)
