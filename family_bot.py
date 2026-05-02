"""Family bot plugin for zoozl server.

Provides a conversational assistant powered by OpenAI for family organisation.

Usage:
    Start with: python -m zoozl --conf family-bot.toml
"""

import logging

import openai

from zoozl.chatbot import Interface


log = logging.getLogger(__name__)


class FamilyAssistant(Interface):
    """Family assistant powered by OpenAI.

    Handles all incoming messages through OpenAI chat completions and
    maintains conversation history per user session.
    """

    aliases = {"greet", "help", "cancel"}

    def load(self, root):
        """Load configuration and initialise the OpenAI async client.

        Configuration is read from the ``[family_bot]`` section of the
        zoozl TOML config file.  ``openai_api_key`` is optional: when
        omitted the client falls back to the ``OPENAI_API_KEY``
        environment variable.
        """
        conf = root.conf.get("family_bot", {})
        api_key = conf.get("openai_api_key")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._model = conf.get("model", "gpt-4o-mini")
        self._system_prompt = conf.get(
            "system_prompt",
            (
                "You are a helpful family assistant. "
                "You help organise family matters such as schedules, tasks, "
                "shopping lists, and family communications."
            ),
        )

    async def consume(self, package):
        """Forward the latest message to OpenAI and send the reply back.

        Conversation history is stored in ``package.conversation.data``
        so that context is preserved across multiple exchanges with the
        same user.
        """
        if "history" not in package.conversation.data:
            package.conversation.data["history"] = []

        history = package.conversation.data["history"]
        user_text = package.last_message_text
        history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": self._system_prompt}] + history

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            reply = response.choices[0].message.content
        except openai.OpenAIError as exc:
            log.error("OpenAI API error: %s", exc)
            reply = "Sorry, I am unable to respond right now. Please try again later."

        history.append({"role": "assistant", "content": reply})
        package.callback(reply)
