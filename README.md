# Steel Plate Fault Detection using Machine Learning

## 📌 Project Overview

Steel plate manufacturing can develop different types of surface faults during the production process. Detecting these faults automatically can help improve quality control and reduce manual inspection.

This project uses **Machine Learning** to detect multiple types of faults in steel plates based on numerical features describing the plate's geometry, luminosity, edges, and other properties.

The project implements **Multi-Label Classification** using **XGBoost with ClassifierChain** and deploys the trained model using **Streamlit**.

---

## 🎯 Objective

The main objective of this project is to:

* Analyze steel plate features.
* Identify important features affecting fault detection.
* Build a multi-label classification model.
* Predict multiple faults simultaneously.
* Evaluate the model using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* Deploy the trained model using Streamlit.

---

## 📊 Dataset

The dataset contains:

* **1,948 records**
* **27 independent features**
* **7 target fault labels**

### Input Features

The model uses 27 features related to steel plate characteristics, including:

* X_Minimum
* X_Maximum
* Y_Minimum
* Y_Maximum
* Pixels_Areas
* X_Perimeter
* Y_Perimeter
* Sum_of_Luminosity
* Minimum_of_Luminosity
* Maximum_of_Luminosity
* Length_of_Conveyer
* TypeOfSteel_A300
* TypeOfSteel_A400
* Steel_Plate_Thickness
* Edges_Index
* Empty_Index
* Square_Index
* Outside_X_Index
* Edges_X_Index
* Edges_Y_Index
* Outside_Global_Index
* LogOfAreas
* Log_X_Index
* Log_Y_Index
* Orientation_Index
* Luminosity_Index
* SigmoidOfAreas

### Target Faults

The model predicts the following 7 fault types:

1. Pastry
2. Z_Scratch
3. K_Scatch
4. Stains
5. Dirtiness
6. Bumps
7. Other_Faults

---

## 🔍 Problem Type

This project uses **Multi-Label Classification**.

Unlike multiclass classification, where one sample belongs to only one class, multi-label classification allows a sample to have multiple labels.

For example:

```text
Pastry       = 1
Z_Scratch    = 0
K_Scatch     = 1
Stains       = 0
Dirtiness    = 0
Bumps        = 1
Other_Faults = 0
```

Therefore, multiple faults can be predicted for the same steel plate.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Pickle

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Analysis
   ↓
Train-Test Split
   ↓
XGBoost Classifier
   ↓
ClassifierChain
   ↓
Multi-Label Prediction
   ↓
Model Evaluation
   ↓
Save Model as model.pkl
   ↓
Streamlit Deployment
```

---

## 🤖 Machine Learning Model

### XGBoost

XGBoost is a gradient boosting algorithm based on decision trees. It is used as the base classifier in this project.

Example configuration:

```python
xgb_model = xgb.XGBClassifier(
    n_estimators=140,
    max_depth=3,
    learning_rate=0.045,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.9,
    gamma=0.15,
    reg_alpha=0,
    reg_lambda=2,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

### ClassifierChain

Since the project contains 7 target labels, `ClassifierChain` is used to handle the multi-label classification problem.

```python
from sklearn.multioutput import ClassifierChain

model = ClassifierChain(
    xgb_model,
    order=None,
    cv=None
)

model.fit(x_train, y_train)
```

---

## 📈 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

For the multi-label problem, `multilabel_confusion_matrix` can be used to obtain a separate 2×2 confusion matrix for each fault label.

---

## 💾 Model Saving

The trained model, feature names, and target names are saved together in `model.pkl`.

```python
import pickle

model_data = {
    "model": model,
    "feature_names": x.columns.tolist(),
    "fault_names": y.columns.tolist()
}

with open("model.pkl", "wb") as file:
    pickle.dump(model_data, file)
```

---

## 🌐 Streamlit Deployment

The trained model is deployed using Streamlit.

The application allows users to enter the 27 feature values and receive predictions for the 7 possible steel plate faults.

### Run the application locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 📁 Project Structure

```text
Steel-Fault-Detection/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
└── steel-faults.csv
```

### File Description

| File               | Description                             |
| ------------------ | --------------------------------------- |
| `app.py`           | Streamlit application                   |
| `model.pkl`        | Trained XGBoost + ClassifierChain model |
| `requirements.txt` | Required Python libraries               |
| `README.md`        | Project documentation                   |
| `steel-faults.csv` | Dataset                                 |

---

## 📦 Requirements

The required libraries are listed in `requirements.txt`:

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
```

---

## 🚀 Future Improvements

* Improve model performance through hyperparameter tuning.
* Add interactive data visualizations.
* Add CSV file upload functionality.
* Display prediction probabilities.
* Add ROC curves for individual fault types.
* Improve the Streamlit user interface.
* Deploy the application online.

---

## 👨‍💻 Author

**Devendra Veera Sudheer**

GitHub: `devendraveerasudheer`

---

## ⭐ Conclusion

This project demonstrates an end-to-end Machine Learning workflow for **steel plate fault detection**, starting from data analysis and feature evaluation to multi-label classification and Streamlit deployment.

The combination of **XGBoost and ClassifierChain** enables the system to predict multiple steel plate faults from the available input features.
# Steel-Fault-Detection
