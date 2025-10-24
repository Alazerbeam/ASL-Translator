# Project Overview

This project builds an end-to-end deep learning model to recognize American Sign Language (ASL) alphabet signs from images. It includes data preprocessing, model training, and evaluation, with full reproducibility via DVC and PyTorch.

# Objectives

- Classify static ASL alphabet signs (A--Z, plus special signs: del, space, nothing)
- Apply image preprocessing and data augmentation
- Use YOLOv8 architecture
- Track datasets and models with DVC for reproducibility
- Future real-time webcam inference

# Project Structure

# Installation

\# Clone the repository
`git clone https://github.com/Alazerbeam/ASL-Translator.git`
`cd ASL-Translator`

\# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   \# On Windows: .\.venv\Scripts\activate

\# Install dependencies
pip install -r requirements.txt

\# Retrieve dataset via DVC
dvc pull

# Dataset

### Source: ASL Alphabet Dataset (Kaggle)
The dataset contains labeled images of ASL alphabet signs and three additional classes: Nothing, Space, and Delete.

# Model Architecture
The project fine-tunes a pretrained YOLOv8 classification model on the ASL alphabet dataset. YOLO’s convolutional backbone and fast inference make it suitable for real-time ASL recognition.

# Results
To be added after initial training.

Planned metrics:
- Accuracy
- F1-score
- Confusion matrix visualization

# Reproducibility with DVC
Data and pipeline versioning are handled via DVC.

To reproduce the preprocessing and training steps:
`dvc repro`

To fetch the exact dataset and model used in this version:
`dvc pull`

# Future Work

- Add webcam-based live ASL letter recognition
- Extend to dynamic signs
- Deploy model
- Experiment with transfer learning for improved accuracy

# Credits
- Dataset: ASL Alphabet Dataset by grassknoted on Kaggle. Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Built with PyTorch, OpenCV, and DVC

# License
This project is licensed under the MIT License.