from setuptools import find_packages, setup

VERSION="0.0.1"
setup(
    name="quaff",
    version=VERSION,
    description='quaff: Quantized parameter-efficient fine-tuning framework',
    license='MIT',
    packages=find_packages(),
    install_requires = [
        "torch",
        "safetensors",
        "bitsandbytes==0.41.0",
        "scipy",
        "wandb",
        "transformers==4.41.2",
        "tqdm",
        "packaging",
        "pytest",
        "numpy",
        "pyyaml",
        "datasets",
        "psutil",
        "setuptools",
        "evaluate",
        "peft",
        "scikit-learn",
        "accelerate==1.1.0",
        "huggingface-hub==0.28.1",
    ]
)