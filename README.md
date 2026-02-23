# 🐒 PrototypicalFewShot: Research-Grade Primate Classification

**Empowering AI with Minimal Data: A Scalable Framework for Few-Shot Image Classification.**

---

## 🧬 Scientific Introduction

This repository provides an institutional-grade implementation of **Prototypical Networks**, a metric-based meta-learning architecture specifically optimized for **Few-Shot Learning (FSL)**. Our framework addresses the challenging task of identifying distinct primate species (e.g., *Indri Indri*, *Diadema Sifaka*) using extremely sparse datasets—relying on as few as five image samples per category.

Unlike conventional deep learning regimes that mandate massive data volumes, this system utilizes a high-dimensional embedding space to generate internal class "prototypes," allowing for high-accuracy inference in biodiversity monitoring and research scenarios where data is inherently limited.

## 💻 Terminal Execution (CLI Usage)

The project is structured as a modular Python package, enabling seamless terminal integration.

### Installation
```bash
# Register the package for development
pip install -e .
```

### Episodic Training
Initiate a training session with configurable episodes and data throughput:
```bash
python src/train.py --data data_lemur/train --episodes 50 --batch-size 32
```

### Automated Inference
Process single imagery or evaluate entire directory structures:
```bash
# Single image diagnostic
python src/predict.py --query path/to/sample.jpg --model-path primate_model.pth

# Batch directory processing
python src/predict.py --query data_lemur/test/ --model-path primate_model.pth
```

## 📓 Theoretical Walkthrough (Interactive Demo)

For researchers seeking a visual intuition of the embedding space and episodic paradigms, the **[LemurProtoType.ipynb](LemurProtoType.ipynb)** serves as an end-to-end tutorial. It implements the identical logic as our production-grade CLI scripts but prioritizes live visualization and categorical breakdown of the N-way K-shot methodology.

## 📊 Portfolio Summary

*   **Meta-Learning Proficiency**: Leveraged Prototypical Networks to achieve high-precision classification in specialized **N-way K-shot benchmarks**.
*   **Modular Neural Architecture**: Engineered a decoupled system for **deep feature extraction**, prototype synthesis, and metric inference.
*   **Production-Grade Tooling**: Developed a comprehensive CLI ecosystem for robust **episodic training** and batch inference workflows.

---
**Advancing ecological research through deep metric learning.**
