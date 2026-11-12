# Heart Disease Classification

**Machine Learning** project that aims to predict whether a patient has **heart disease** based on their clinical parameters.

> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

This is a **binary classification** problem: each patient is labeled as `1` (heart disease) or `0` (no heart disease).

## Dataset

The project uses the heart disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease), available at `data/heart-disease.csv` (303 patients, 13 features plus the `target` column, no missing values).

| Feature | Description |
|---------|-------------|
| `age` | Age |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thallium stress test result |
| `target` | Presence of heart disease (1/0) |

## Project structure

```
├── Classification.ipynb   # Notebook with the full analysis and modelling
├── data/
│   └── heart-disease.csv  # Dataset
├── images/                # Images used in the notebook
├── LICENSE
└── README.md
```

## Methodology

The notebook follows the 6-step Machine Learning framework:

1. **Problem definition**
2. **Data**
3. **Evaluation**
4. **Features** and exploratory data analysis (EDA): class distribution, heart disease by sex, age vs. maximum heart rate, chest pain type and correlation matrix
5. **Modelling**
   - Baseline model comparison: Logistic Regression, KNN and Random Forest
   - Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`
   - Evaluation: ROC curve / AUC, confusion matrix, classification report, cross-validated metrics and feature importance
6. **Experimentation**

## Results

Test set accuracy (80 % train / 20 % test split):

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 88.5 % |
| Random Forest | 83.6 % (86.9 % after `RandomizedSearchCV`) |
| KNN | 68.9 % (75.4 % with the best `n_neighbors`) |

The best model is **Logistic Regression** with `C ≈ 0.204` and `solver="liblinear"` (found with `GridSearchCV`).

Metrics using 5-fold cross-validation:

| Metric | Mean |
|--------|------|
| Accuracy | 84.5 % |
| Precision | 82.1 % |
| Recall | 92.1 % |
| F1-score | 86.7 % |

Possible next steps include collecting more data and trying models such as CatBoost or XGBoost.

## Tech stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Getting started

```bash
git clone https://github.com/rafaroman18/Heart-Disease-Classification.git
cd Heart-Disease-Classification
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Classification.ipynb
```

> The notebook uses `plot_roc_curve`, which was removed in scikit-learn 1.2. To run it as is, use `scikit-learn<1.2`, or replace it with `RocCurveDisplay.from_estimator`.

## License

Distributed under the license included in the [LICENSE](LICENSE) file.
