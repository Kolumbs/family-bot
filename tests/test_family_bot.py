"""Tests for the family_bot plugin."""

import sys
import os
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the project root is on the path so family_bot can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import family_bot


class TestFamilyAssistantLoad(IsolatedAsyncioTestCase):
    """Tests for the FamilyAssistant.load method."""

    def _make_root(self, conf=None):
        """Return a minimal mock InterfaceRoot."""
        root = MagicMock()
        root.conf = conf or {}
        return root

    @patch("family_bot.openai.AsyncOpenAI")
    def test_load_defaults(self, mock_openai_cls):
        """Plugin loads with default settings when no config section present."""
        assistant = family_bot.FamilyAssistant()
        assistant.load(self._make_root())

        mock_openai_cls.assert_called_once_with(api_key=None)
        self.assertEqual(assistant._model, "gpt-4o-mini")
        self.assertIn("family assistant", assistant._system_prompt)

    @patch("family_bot.openai.AsyncOpenAI")
    def test_load_custom_config(self, mock_openai_cls):
        """Plugin picks up custom model and system_prompt from config."""
        conf = {
            "family_bot": {
                "openai_api_key": "sk-test",
                "model": "gpt-4o",
                "system_prompt": "Custom prompt",
            }
        }
        assistant = family_bot.FamilyAssistant()
        assistant.load(self._make_root(conf))

        mock_openai_cls.assert_called_once_with(api_key="sk-test")
        self.assertEqual(assistant._model, "gpt-4o")
        self.assertEqual(assistant._system_prompt, "Custom prompt")

    def test_aliases(self):
        """FamilyAssistant handles ordinary messages via greet/help.

        It must NOT claim the ``cancel`` alias: zoozl invokes the cancel
        handler as a pre-check on every turn, so claiming it would run the
        OpenAI consume twice and send a duplicate reply per message.
        """
        self.assertIn("greet", family_bot.FamilyAssistant.aliases)
        self.assertIn("help", family_bot.FamilyAssistant.aliases)
        self.assertNotIn("cancel", family_bot.FamilyAssistant.aliases)


class TestFamilyAssistantConsume(IsolatedAsyncioTestCase):
    """Tests for the FamilyAssistant.consume method."""

    def _make_assistant(self):
        """Return a loaded FamilyAssistant with a mocked OpenAI client."""
        assistant = family_bot.FamilyAssistant()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI reply"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        assistant._client = mock_client
        assistant._model = "gpt-4o-mini"
        assistant._system_prompt = "You are helpful."
        return assistant, mock_client

    def _make_package(self, user_text="Hello"):
        """Return a minimal mock Package."""
        package = MagicMock()
        package.last_message_text = user_text
        package.conversation.data = {}
        package.callback = MagicMock()
        return package

    async def test_sends_reply_to_callback(self):
        """consume() passes the OpenAI reply to package.callback."""
        assistant, _ = self._make_assistant()
        package = self._make_package("Hello")

        await assistant.consume(package)

        package.callback.assert_called_once_with("OpenAI reply")

    async def test_history_built_correctly(self):
        """User and assistant messages are appended to conversation history."""
        assistant, mock_client = self._make_assistant()
        package = self._make_package("What is 2+2?")

        await assistant.consume(package)

        history = package.conversation.data["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "What is 2+2?"})
        self.assertEqual(history[1], {"role": "assistant", "content": "OpenAI reply"})

    async def test_system_prompt_included(self):
        """System prompt is prepended to the messages sent to OpenAI."""
        assistant, mock_client = self._make_assistant()
        package = self._make_package("Hi")

        await assistant.consume(package)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "You are helpful."})

    async def test_history_persists_across_turns(self):
        """Conversation history grows with each call."""
        assistant, mock_client = self._make_assistant()
        package = self._make_package("First message")

        await assistant.consume(package)

        # Simulate a second turn
        package.last_message_text = "Second message"
        await assistant.consume(package)

        history = package.conversation.data["history"]
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[2]["role"], "user")
        self.assertEqual(history[3]["role"], "assistant")

    async def test_openai_error_returns_fallback_message(self):
        """When OpenAI raises an error a friendly fallback is returned."""
        import openai as openai_mod

        assistant, mock_client = self._make_assistant()
        mock_client.chat.completions.create.side_effect = openai_mod.OpenAIError(
            "network error"
        )
        package = self._make_package("Hello")

        await assistant.consume(package)

        text = package.callback.call_args.args[0]
        self.assertIn("unable to respond", text)
