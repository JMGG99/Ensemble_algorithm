# Cancer Classification with VotingClassifier

## Overview
This project applies ensemble learning using the **VotingClassifier** to classify cancer diagnoses based on medical features. The dataset used contains information about tumor characteristics, and the goal is to predict whether a tumor is malignant or benign.

## Dataset
The dataset is obtained from a publicly available source and includes features such as:
- **radius_mean**
- **concave points_mean**
- **diagnosis (Malignant/Benign)**

Dummy variables are created for categorical data, and only relevant numerical features are used for classification.

## Models Used
The classification models included in the ensemble are:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier**

These models are combined using **VotingClassifier** to improve overall prediction performance.

## Evaluation Metrics
The following performance metrics are used to assess the models:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```
2. Run the script to train and evaluate the models:
   ```bash
   python cancer_classification.py
   ```

## Results
Each model is trained and evaluated individually, and then the **VotingClassifier** is applied to combine predictions. The results are displayed in terms of accuracy, precision, recall, and F1-score.

## License
This project is for educational purposes and does not include a specific license.

