# <img src="figures/beers.png" width="50"> Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis

This repository contains the code for the paper "Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis."

## Overview
We first the Outlier Spatial Stability Hypothesis (**OSSH**): *During fine-tuning, certain activation outlier channels retain stable spatial position across training iterations*. 

Building on OSSH, we propose **Quaff**, a **Qua**ntized parameter-e**f**ficient **f**ine-tuning framework for LLMs that decouples weight and activation quantization via targeted momentum scaling. Quaff dynamically computing scaling factors exclusively for invariant outlier channels, eliminating full-precision weight storage and global rescaling, enabling low quantization error and high efficiency.

## Installation
To install Quaff, just use

```
conda create -n quaff python=3.10
conda activate quaff
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset and Model
Quaff supports fine-tuning on ten datasets and three models. All datasets and models are available on Hugging Face and will be automatically downloaded if the ```transformers``` package is properly installed.
The dataset and corresponding model will be automatically downloaded when you run the fine-tuning task.

## Run the experiment
### Quick start
To perform LoRA fine-tuning on the OIG/chip dataset using the Phi3-3.8B model with Quaff, run the following command:
```
bash script/peft/lora/run_quaff_phi3_chip.sh
```


### Predefining Outlier Channels
If you wish to predefine the outlier channels for a model, refer to the scripts in the```script/generate_outlier/```directory. For example, to predefine outlier channels for the Phi-3 model, run:
```
bash script/generate_outlier/run_phi3.sh
```