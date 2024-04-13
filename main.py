import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import os
from huggingface_hub import login


# Define the request body data structure
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 2000 

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initializes and loads the tokenizer and language model upon application startup.
    
    The function is executed when the application is starting up. It is used to setup the tokenizer 
    and the model for receiving requests. The function also retrives the token which is ussed to login.
    
    Globals:
        tokenizer (AutoTokenizer): A tokenizer for processing input prompts.
        model (AutoModelForCausalLM): A pre-trained language model for generating text.
    """
    global tokenizer, model
    hf_token = os.getenv("HF_TOKEN")
    print(hf_token)
    if not hf_token:
        raise ValueError("No Hugging Face token provided")
    login(token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        torch_dtype=torch.bfloat16
    )


@app.post("/generate/")
async def generate_text(request: GenerationRequest):
    """
    Generates response based on the provided essay prompt to evaluate whether it was
    composed by AI or a human.
    
    This endpoint accepts a request containing an essay prompt and uses the pre-loaded
    model to generate an evaluation of the essay. 
    
    Args:
        request (GenerationRequest): Request object containing the 'prompt' and
                                     'max_length' fields

    Returns:
        dict: A response returning to the requesting client

    Raises:
        HTTPException: An error response with status code 500 if there is an issue during
                       the response generation process.
    """
    try:
        prompt = f"{request.prompt} \nPlease evaluate the following essay and determine if it was composed by AI or a human: "
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=request.max_length)
        text = tokenizer.decode(outputs[0])
        # Split the text to remove the initial repeated prompt
        parts = text.split("Please evaluate the following essay and determine if it was composed by AI or a human:")
        response_text = parts[1].strip() if len(parts) > 1 else "No evaluation found."
        return {"result": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
