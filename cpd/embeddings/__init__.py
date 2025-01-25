from cpd.embeddings.prompts import ComplexPrompt, CompositionalPrompt
from cpd.embeddings.transforms import *


def get_prompt_map(self, text, tokenizer):        
    tokenized = tokenizer(text,  
                        truncation=True, 
                        max_length=77, 
                        return_length=True,
                        return_overflowing_tokens=False, 
                        padding="max_length", 
                        return_tensors="pt")
    token_ids = tokenized["input_ids"].squeeze().cpu()
    return [tokenizer.decode(id) for id in token_ids]
