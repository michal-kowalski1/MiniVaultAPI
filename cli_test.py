import requests
import sys
import json
import os
from datetime import datetime, timezone

API_URL_FULL = "http://127.0.0.1:8000/generate"
API_URL_STREAM = "http://127.0.0.1:8000/generate-stream"
LOG_FILE = "log.jsonl"

def log_interaction(prompt, response, mode="full"):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "prompt": prompt,
        "response": response
    }
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/{LOG_FILE}", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

def send_prompt_full(prompt):
    try:
        response = requests.post(API_URL_FULL, json={"prompt": prompt})
        response.raise_for_status()
        result = response.json()
        reply = result.get("response", "[No response]")
        print("Full Response:\n", reply)
        log_interaction(prompt, reply, mode="full")
    except Exception as e:
        print("Error:", e)
        log_interaction(prompt, f"[ERROR: {e}]", mode="full")

def send_prompt_stream(prompt):
    try:
        with requests.post(API_URL_STREAM, json={"prompt": prompt}, stream=True) as response:
            response.raise_for_status()
            print("Streaming Response:")
            streamed_output = ""
            for chunk in response.iter_content(chunk_size=1):
                token = chunk.decode("utf-8")
                print(token, end="", flush=True)
                streamed_output += token
            print()
            log_interaction(prompt, streamed_output, mode="stream")
    except Exception as e:
        print("Streaming Error:", e)
        log_interaction(prompt, f"[ERROR: {e}]", mode="stream")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python cli_test.py 'Your prompt'")
        print("  python cli_test.py --stream 'Your prompt'")
    else:
        args = sys.argv[1:]
        if args[0] == "--stream":
            prompt = " ".join(args[1:])
            send_prompt_stream(prompt)
        else:
            prompt = " ".join(args)
            send_prompt_full(prompt)