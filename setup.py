from setuptools import setup, find_packages

setup(
    name="chemglam",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.0",
        "torchvision==0.20.0",
        "torchaudio==2.5.0",
        "transformers==4.46.3",
        "lightning==2.4.0",
        "peft==0.13.2",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "wandb>=0.12.0"
    ],
)
