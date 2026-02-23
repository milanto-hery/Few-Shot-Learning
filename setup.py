from setuptools import setup, find_packages

setup(
    name="prototypical_fewshot",
    version="1.0.0",
    description="Research-grade Prototypical Networks for Few-Shot Image Classification",
    author="Research Engineer",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pillow",
        "matplotlib",
        "pandas",
        "numpy",
    ],
    python_requires=">=3.7",
)
