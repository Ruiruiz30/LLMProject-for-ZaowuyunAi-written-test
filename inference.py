import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"D:\LLMProject\Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

input_text = "你好！你是谁？叫什么名字？"

input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    output = model.generate(input_ids, max_length=100)  # max_length 可根据需要调整

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)