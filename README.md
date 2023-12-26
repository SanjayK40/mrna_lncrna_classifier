# mRNA and lncRNA Classification Model

## Overview
This project implements a classification model for distinguishing between lncRNA and mRNA sequences based on k-mers and a Convolutional Neural Network (CNN).

## Research Paper
Title: A classification model for lncRNA and mRNA based on k-mers and a convolutional neural network.

## Methodology
### Determination of k-mer parameters
1. **Data Selection:**
   - lncRNA sequences (250 nt to 3500 nt)
   - mRNA sequences (200 nt to 4000 nt)
2. **K-mer Extraction:**
   - Extracted using the k-mer algorithm.
   - For k-mers with larger values, relative entropy is used for feature selection.
   - Frequency of k-mer subsequences is counted to construct a frequency matrix.
3. **Convolutional Neural Network:**
   - Trained using the constructed frequency matrix.
   - CNN architecture:
     - First layer: 32 convolution kernels of 3 × 3, Relu activation function.
     - Second layer: 64 convolution kernels of 3 × 3, Relu activation function.
     - Third layer: Largest pooling layer (2 × 2).
     - Dropout to prevent overfitting.
     - Last layer: Fully connected layer with 128 neurons, SoftMax function for classification.
   - Loss function: Cross entropy, Optimizer: Adadelta.

### Determination of optimal k-mer combination
1. **Dataset:**
   - 4556 lncRNA sequences (250 nt to 3000 nt) from GENCODE database.
   - 4556 mRNA sequences (200 nt to 4000 nt).
2. **Model Training:**
   - CNN model trained on selected sequences.
   - Differential k-mers selected with different k values.
   - Evaluation metrics: Classification accuracy, model accuracy, recall rate, and F1 score.

## Code Structure
- `main.ipynb`: Jupyter notebook containing the main code.
- `utils.py`: Utility functions.
- `data/`: Directory containing dataset files.
- `models/`: Directory for saving trained models.
- `results/`: Directory for storing evaluation results.

## Installation
```bash
pip install -r requirements.txt
