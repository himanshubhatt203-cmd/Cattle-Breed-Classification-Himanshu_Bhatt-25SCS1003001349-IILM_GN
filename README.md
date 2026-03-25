# Cattle Breed Classification

## Overview
This project is a deep learning-based system designed to classify different cattle breeds from images. It uses a trained convolutional neural network (CNN) model built with PyTorch to accurately predict the breed of cattle from an input image. The project also includes a graphical user interface (GUI) for easy interaction and demonstration.

## Objective
The aim of this project is to apply machine learning techniques to solve a real-world agricultural problem by automating the identification of cattle breeds, which can assist in livestock management, research, and productivity improvement.

## Features
- Image-based cattle breed prediction
- Pre-trained deep learning model
- GUI interface for user-friendly interaction
- Confidence score display
- Label mapping using JSON
- Modular code structure (training, evaluation, prediction)
- Data preprocessing and training pipeline included

## Technologies Used
- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- PyQt6 / Tkinter (GUI)

## Project Structure
Cattle_Breed_Classification/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── labels_json.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── models/
│   ├── best_model.pth
│   └── labels.json
│
├── test/
│   └── sample images
│
├── data_preprocessing/
│   ├── Data_Splitting.py
│   ├── Data_Cleaning.py
│   ├── Feature_space_Viz.py
│   ├── Model_training.py
│   ├── Model_initialization.py
│   ├── Data_loading.py
│   ├── Splitting_data.py
│   ├── Data.py
│   ├── Detailed_Transforms.py
│   ├── Cuda_check.py
│   ├── main.py
│   └── import_libs.py
│
├── docs/
│   ├── AIML_Report.docx
│   └── project_presentation.pptx
│
├── final_ui.py
├── main.py
├── README.md
├── requirements.txt
├── .gitignore

## How It Works
1. Input image is preprocessed (resized and normalized)
2. The trained model predicts the cattle breed
3. Output is displayed with prediction confidence
4. GUI allows users to easily test images

## How to Run
1. Install requirements
2. Run final_ui.py
3. Upload/select image
4. Get predicted breed

## Model Files
- best_model.pth
- labels.json
- 
## Dataset
Dataset is not included in this repository due to large size.

## Dataset Setup
To train the model, create this folder structure in the project root:
dataset/
├── train/
├── val/
└── test/

## Output
The model predicts one of multiple cattle breeds and displays the result along with confidence score.

## Applications
- Livestock management
- Agricultural research
- Breed identification automation
- Educational and academic purposes

## Future Improvements
- Improve model accuracy with larger datasets
- Deploy as a web/mobile application
- Add real-time camera-based prediction
- Integrate cloud-based model serving

## Documentation
Project report and presentation are available in the docs folder.
