from transformers import (
    AutoModelForCausalLM,
)
import torch 
import quaff
from quaff.nn.model import build_quantized_model
from peft import (
    LoraConfig,
    get_peft_model,
)
import numpy as np
from transformers import AutoTokenizer
import time

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_auth_token=True,).to("cuda:0")

outlier_channels = torch.load("outlier/mean_times100.0_chip2_Phi-3-mini-4k-instruct.pt")

# if you want ot compare the speed of quaff with FP32, simply comment following quantization step 
model = build_quantized_model(model, outlier_channels)
print("############ Quantized model structure ############")
print(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules= "all-linear",
    lora_dropout=0.1,
    bias="none",
    inference_mode=False,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print("############ model structure after setting peft ############")

example = "We propose the Outlier Spatial Stability Hypothesis - during fine-tuning, activation outlier channels maintain stable spatial positions across training iterations. Based on this observation, Quaff enables efficient LLM adaptation through: 1. Targeted Momentum Scaling: Dynamic scaling factors computed exclusively for stable outlier channels. 2. Decoupled Quantization: Separate weight and activation quantization strategies. 3. Hardware Efficiency: Eliminates full-size full-precision weight storage and global rescaling"

tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
tokens = tokenizer(example, truncation=True, padding='max_length', max_length=512)

inputs = {k:torch.tensor(v).unsqueeze(0).to("cuda:0") for k, v in tokens.items()}

model.to("cuda:0")
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
start_time = time.time()
outputs = model(input_ids=inputs["input_ids"][...,:-1], labels=inputs["input_ids"][...,1:])
fw_time = time.time()
loss = outputs[0]
loss.backward()
optimizer.step()
bw_time = time.time()

print(f"forward time: {fw_time - start_time}, backward time: {bw_time - fw_time}, all time {bw_time - start_time}")