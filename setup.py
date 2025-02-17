from setuptools import find_packages, setup

VERSION="0.0.1"
setup(
    name="quaff",
    version=VERSION,
    description='quaff: Quantized parameter-efficient fine-tuning framework',
    license='MIT',
    packages=find_packages(),
)