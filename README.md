# Cattle Breed Classification

## Overview
This project is a deep learning based cattle breed classification system developed using Python and PyTorch. It predicts the breed of cattle from an input image and also provides a GUI-based interface for easy use.

## Features
- Image-based cattle breed prediction
- Trained deep learning model
- GUI interface
- Label mapping using JSON
- Test images included

## Technologies Used
- Python
- PyTorch
- torchvision
- scikit-learn
- NumPy
- PyQt6

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

## How to Run
1. Install requirements
2. Run final_ui.py
3. Upload/select image
4. Get predicted breed

## Model Files
- best_model.pth
- labels.json

## Dataset
Dataset is not included in this repository due to large size.

## Documentation
Project report and presentation are available in the docs folder.
