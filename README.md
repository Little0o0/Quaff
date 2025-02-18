# <img src="figures/beers.png" width="40"> Quaff: Quantized Language Model Adaptation Framework

Efficiently fine-tuning large language models (LLMs) with Quaff — a framework designed for optimized performance without sacrificing accuracy. Leverage quantization techniques to reduce latency/memory usage while maintaining high performance.

## 🚀 Key Features
- **Targeted Momentum Scaling**: Dynamic scaling factors computed exclusively for stable outlier channels.
- **Decoupled Quantization**: Independent weight and activation quantization strategies.
- **Hardware Efficiency**: Eliminates full-size FP weight storage and global rescaling.
- **Performance Boosts**: Significant speed improvements without accuracy loss.


## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quaff.git

## create envs
conda create -n quaff python=3.10
conda activate quaff

# Install dependencies
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 📚 Dataset and Model
Quaff supports ```HuggingFace``` and provides tools for seamless integration. All datasets and models are available on HuggingFace and will be automatically downloaded if the ```transformers``` package is properly installed.

## 💻 Quick Start
### Build Your Own Task
To use Quaff for fine-tuning on your own task to achieve amazing acceleration, follow these steps:

1. Load your FP model (we’ll use the Phi-3 model as an example):

```
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    torch_dtype=torch.float32,
    trust_remote_code=True,
    use_auth_token=True,).to("cuda:0")
```

2. Determine outlier channels using a calibration dataset. Refer to ```quaff/calibration/outlier_detection.py```  for details on how to do this. Save the output (e.g., ```mean_times100.0_chip2_Phi-3-mini-4k-instruct.pt```) in the ```outlier/``` directory.

3. Load predefined outlier channels and quantize the model:
```
outlier_channels = torch.load("outlier/mean_times100.0_chip2_Phi-3-mini-4k-instruct.pt")
model = build_quantized_model(model, outlier_channels)
print("############ Model structure after setting PEFT ############")
print(model)
```

4. Inject PEFT parameters (e.g., LoRA fine-tuning):
```
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
```
5. Start fine-tuning the model without no extra operation. 
```
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
```

The example is available in ```quick_start.py```

**Performance Comparison.**
On the RTX 5880 GPU, the output of ```quick_start.py``` for FP32 model is 
```
forward time: 0.35955357551574707, backward time: 0.20281243324279785, all time 0.5623660087585449
```
The output for Quaff model is 
```
forward time: 0.22443771362304688, backward time: 0.11957168579101562, all time 0.3440093994140625
```
achieving $1.65 \times$ speedup.

> This one-step latency has limited reference value due to the GPU's slow startup, but it still provides some insight into efficiency of Quaff. To accurately evaluate efficiency, the model should be run for multiple steps, and the average latency should be calculated, as shown in our paper.

### Running the Bash Scripts
If you want to simply use our supported models and datasets, you can run the experiments usign the bash files. 
First, to predefine the outlier channels for a model, you can use the scripts in the```script/generate_outlier/```directory. For example, to predefine outlier channels for the Phi-3 model, run:
```
bash script/generate_outlier/run_phi3.sh
```
The default outlier saving directory is ```./outlier/```

To perform LoRA fine-tuning on the OIG/chip2 dataset using the Phi-3 3.8B model with Quaff, use the following command:
```
bash script/peft/lora/run_quaff_phi3_chip.sh
```

The details about the arguments are provided in ```.\utils\arguments.py```.
