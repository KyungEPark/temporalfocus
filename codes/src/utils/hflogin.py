from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import torch
from huggingface_hub import login

def hflogin():
    login("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")
    access_token = "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"
    