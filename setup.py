from setuptools import setup, find_packages

setup(
    name="voiceai",  # Replace with your project name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "vllm==0.7.2",
        "transformers==4.46.2",
        "deepspeed==0.16.4",
        "pydantic==2.10.6",
        "coqui-tts==0.25.3",
    ],
    python_requires=">=3.10",
)
