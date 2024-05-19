from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from src.utils.finetune_preprocess import makeexamples, formatting_prompts_func
'''
# Make sure to create the args before running
## Example: 
args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 60,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407, 
            report_to="wandb"
        )
'''

def llama3_unsloth(bertn, liwcn, args, output_path):
    max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!    
    # Bringing model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    # Call data
    bertpath = r'data/rawdata/finalberttrain.pkl'
    bertdf = berttopn(bertpath, bertn)
    bertex = preprocess(bertdf)
    liwcpath = r'data/rawdata/finalliwctrain.pkl'
    liwcdf = liwctopn(liwcpath, liwcn)
    liwcex = preprocess(liwcdf)
    examplelist = bertex + liwcex
    examples = '. '.join(examplelist)
    traindata = makeexamples(examples)    
    dataset = load_dataset(traindata, split = "train")
    dataset = dataset.map(lambda example:formatting_prompts_func(tokenizer, example), batched = True,)


    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = True, # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        tokenizer = tokenizer,
        args = args,
    )
    trainer.train()
    torch.save(model.state_dict(), output_path)

