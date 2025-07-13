from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

app = FastAPI()

#------------------------------------Initial Settings-----------------------------------------#

MODEL_NAME = "distilgpt2"  # I choose this model. Speed is fast, but the accuracy is low. To improve the perfermance, We can use "EleutherAI/gpt-neo-1.3B" or "openai-community/gpt2".
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

LOG_PATH = "logs/log.jsonl"
os.makedirs("logs", exist_ok=True)

class PromptRequest(BaseModel):
    prompt: str


#------------------------------------Log Function-----------------------------------------#

def log_interaction(prompt, response):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "response": response
        }) + "\n")

#---------------------------------------API-------------------------------------------------#

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    response = generate_full(prompt)
    log_interaction(prompt, response)
    return {"response": response}

@app.post("/generate-stream")
async def generate_stream(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    return StreamingResponse(token_stream(prompt), media_type="text/plain")


#----------------------------Main Processing Function--------------------------------------#
def generate_full(prompt):      # Sends a full prompt and receives a complete response.
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.85,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def token_stream(prompt):       # Returns token-by-token streaming output (simulated from Hugging Face .generate()).
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    max_new_tokens = 60
    generated = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            generated = torch.cat((generated, next_token), dim=1)

            decoded = tokenizer.decode(next_token[0])
            yield decoded
            if decoded.strip() in [".", "!", "?"]:
                break