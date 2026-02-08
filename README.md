# 🐒 PrimatePrototypicalNet: 5-Shot Species Classifier
This repository implements a Prototypical Network for the few-shot classification of primate species. Using only 5 images per category, the model learns to identify complex species like the Indri Indri and the Diadema Sifaka with high precision.

## The Methodology: Few-Shot Learning
Traditional Deep Learning requires thousands of images. This project uses Prototypical Networks, which work by:

Embedding: Transforming images into a 512-dimensional feature space using a ResNet18 backbone.

Prototypes: Calculating the "average" vector (the prototype) for each species based on the 5-shot support set.

Distance-Based Inference: Using Euclidean Distance to compare a new query image to the prototypes. The shortest distance determines the classification.

## Dataset & Annotation
This project uses Implicit Folder Annotation. No external CSV or XML labels are required.

Support Set: Located in data_lemur/train/. Each subfolder name (e.g., lemur_catta) acts as the label.

Query Set: Located in data_lemur/test/. Used to evaluate the model's generalization.
