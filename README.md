# family-bot

Helps organise your family matters.

A WhatsApp chatbot that uses [OpenAI](https://openai.com) to answer questions
and assist with everyday family tasks such as schedules, shopping lists, and
task management.  It is built as a plugin for the
[zoozl](https://github.com/Kolumbs/zoozl) server framework.

## Requirements

- Python 3.11+
- A [Meta / WhatsApp Business](https://developers.facebook.com/docs/whatsapp) account with a configured webhook
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
pip install zoozl openai
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
| `[family_bot].system_prompt` | Custom system prompt for the assistant |

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

1. Registers the `greet`, `help`, and `cancel` aliases so it handles every
   incoming message.
2. Maintains per-user conversation history inside the zoozl `Conversation`
   object so context is preserved across multiple exchanges.
3. Forwards each message to the OpenAI Chat Completions API and sends the
   reply back to the WhatsApp user.
