## MiniVault API — Hugging Face LLM + FastAPI

This is a lightweight local REST API built with **FastAPI** and **Hugging Face Transformers**. It allows us to send a prompt to a local LLM and receive either:

- A full, one-shot response (`/generate`)
- A token-by-token streamed response (`/generate-stream`)
- Command-line interface support (`cli_test.py`)
- JSONL logging for all input/output interactions

---

## Requirements

- Python 3.8+
- pip (Python package manager)
- CPU or GPU (GPU recommended for larger models)

---

## Setup

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run the FastAPI server

uvicorn main:app --reload
This will start the API at http://127.0.0.1:8000.

## Endpoints
- POST /generate
Sends a full prompt and receives a complete response.

Request: { "prompt": "Q: What time is it now? A:" }

- POST /generate-stream
Returns token-by-token streaming output (simulated from Hugging Face .generate()).

Request: { "prompt": "Q: What time is it now? A:" }

## CLI Usage
Run from terminal:

### Full (non-streaming):
  python cli_test.py "Q: What time is it now? A:"

### Streaming:
  python cli_test.py --stream "Q: What time is it now? A:"

## Postman Collection
We can use Postman to manually test the endpoints:

How to Use:
  1. Open Postman
  2. Create a POST request to:
        http://127.0.0.1:8000/generate
  3. Under Body → raw → JSON
        { "prompt": "Q: What time is it now? A:" }

  4. For streaming:
    - Use POST http://127.0.0.1:8000/generate-stream
    - Set Body the same as above
    - Expect streamed text output

## Model Configuration
- The default model is:
MODEL_NAME = "distilgpt2"
- Other options we can use:

  gpt2
  EleutherAI/gpt-neo-1.3B" (requires more memory)
  tiiuae/falcon-7b-instruct" (requires GPU)
  openai-community/gpt2" (safe fallback fork)

- To change the model, just replace the model name in main.py.

## Logs
- All interactions are saved in JSONL format:

    logs/log.jsonl – for API interactions
    logs/cli_log.jsonl – for CLI prompts
- Each line contains:
    {
      "timestamp": "2025-07-13T12:00:00+00:00",
      "mode": "stream",
      "prompt": "Q: What is the capital of France?\nA:",
      "response": "Paris."
    }

## Design Choices & Tradeoffs

  - Streaming was simulated using greedy decoding (argmax) instead of sampling to match Hugging Face’s limitations (they do not support native token-by-token HTTP streaming).
  - For simplicity and reliability, all prompts are logged locally in flat JSONL files. Future improvements could include a web dashboard or log viewer.
  - The model is loaded at server start for performance; if we want per-request flexibility or dynamic loading, lazy loading or background workers could be implemented. And We can use cloud base libraries or big capacity libraries that have trained enough.
  - The API currently uses CPU by default, which is slower for large models. GPU support is seamless if PyTorch detects CUDA.

## Powered By
  - FastAPI
  - Transformers
  - PyTorch
  - Uvicorn

## Notes
  This is a completely local API — no OpenAI, no cloud APIs.
  
  Perfect for offline, privacy-preserving LLM experiments.
  
  Streaming is simulated using greedy token-by-token generation.
