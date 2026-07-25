# family-bot

Helps organise your family matters.

A WhatsApp chatbot that uses the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
to answer questions and assist with everyday family tasks such as schedules,
shopping lists, and task management.  It is built as a plugin for the
[zoozl](https://github.com/Kolumbs/zoozl) server framework.

## Requirements

- Python 3.11+
- A [Meta / WhatsApp Business](https://developers.facebook.com/docs/whatsapp) account with a configured webhook
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
pip install zoozl openai-agents
```

## Configuration

Copy `family-bot.toml` and fill in your credentials:

| Field | Description |
|---|---|
| `whatsapp_port` | Port the webhook server listens on (default `8082`) |
| `whatsapp_verify_token` | Verification token you enter in the Meta dashboard |
| `whatsapp_access_token` | Meta Graph API access token |
| `whatsapp_phone_number_id` | WhatsApp phone number ID from the Meta dashboard |
| `whatsapp_app_secret` | *(optional)* App secret for request signature verification |
| `[family_bot].openai_api_key` | *(optional)* OpenAI API key – defaults to `OPENAI_API_KEY` env var |
| `[family_bot].model` | OpenAI model to use (default `gpt-4o-mini`) |
| `[family_bot].system_prompt` | Custom system prompt (the agent's instructions) |
| `database` | SQLite database used for persisted conversation/session history (default `family_bot_sessions.db`) |
| `[family_bot].history_window` | Max recent items replayed to the model per turn (default `10`) |

You can also pass `openai_api_key`, `whatsapp_access_token`, etc. via environment
variables instead of putting them in the TOML file.

## Running

```bash
python -m zoozl --conf family-bot.toml
```

The server will start listening for WhatsApp webhook notifications on the
configured port.  Point your Meta webhook URL at
`http://<your-server>:<whatsapp_port>/`.

## How it works

`family_bot.py` implements a single `FamilyAssistant` plugin that:

1. Registers the `greet` and `help` aliases so it handles every incoming
   message. (`cancel` is intentionally not claimed — zoozl invokes the cancel
   handler as a pre-check on every turn, so owning it would run the agent twice
   and send a duplicate reply.)
2. Hands each message to a single OpenAI Agents SDK `Agent` whose instructions
   are the configured system prompt.
3. Persists per-user conversation history (keyed by the WhatsApp `wa_id`) in a
   SQLite session store, and replays it to the model through a bounded sliding
   window (`WindowedSession`) so context is preserved across exchanges while the
   prompt never grows without limit. The window is trimmed forward so it always
   starts on a user message.
