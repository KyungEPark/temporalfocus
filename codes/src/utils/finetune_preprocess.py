# Function to make the example
def makeexamples(filepath):
    import json
    import pandas as pd
    import re
    with open('data.json', 'r') as file:
        data = json.load(file)

    # Initialize lists to store data
    instruction = []
    inputs = []
    outputs = []

    pattern = r"The text: '(.+)' has a temporal focus of (.+) "

    # Process each entry in the JSON file
    for entry in data:
        # Extract sentence and label using regex
        match = re.match(pattern, entry)
        if match:
            sentence = match.group(1)
            label = match.group(2)
        
            # Construct instruction, input, and output
            instruction1 = 'You are a chatbot deciphering the temporal focus of a given sentence. You can answer in one of the three words: past, present or future.'
            input_data = sentence
            output_data = label
        
            # Append to lists
            instruction.append(instruction1)
            inputs.append(input_data)
            outputs.append(output_data)

    # Create a DataFrame
    df = pd.DataFrame({'instruction': instructions, 'input': inputs, 'output': outputs})
    return df


def formatting_prompts_func(tokenizer, examples):
    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
'''
# Format to use this:
from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)
'''