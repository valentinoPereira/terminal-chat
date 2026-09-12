## My First ai agent
This is a terminal chat application that talks to the NeuralWatt API.

## Setup

1. Copy `.env.example` to `.env`
2. Set `NEURALWATT_API_KEY` to your NeuralWatt API key
3. If you have a different OpenAI-compatible key, that goes in its own variable — this app only uses `NEURALWATT_API_KEY`

## How to run

1. Open your terminal in this project directory
2. Run `uv run my-first-ai-agent` to start the app
3. Type your message. `/quit` or `Ctrl+C` exits; `/clear` resets the conversation
