from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import torch
from huggingface_hub import login
import json
from src.utils.hflogin import hflogin
import pandas as pd
from tqdm import tqdm
import argparse
import json


# Required

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_id, token = access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
    
def fewshot_reastempfoc(model, tokenizer, sentence, examples):

    
    messages = [
        {"role": "system", "content": "You are a chatbot deciphering the temporal focus\
          of a given sentence. Please provide the reason and label in the format of: \
         'Reason: (Your reason), Answer: (Your answer in one of three words: Past,\
          Present, Future).' Here are some examples:" + examples},
        {"role": "user", "content": "What is the temporal focus of:" + sentence}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=1,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    classifier_str = response.lower()
    return classifier_str

