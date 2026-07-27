# Employee Attrition Prediction using Decision Tree and Random Forest Classification

This project implements machine learning models to predict employee attrition using demographic, professional, and work-related attributes. It evaluates and compares the performance of a **Decision Tree Classifier** and a **Random Forest Classifier** on the IBM HR Analytics Employee Attrition & Performance dataset.

---

## Student Information
- **Name:** Abhilash Choudhary
- **Registration No.:** 23BCE11155
- **Application No.:** IN26012658
- **Batch:** 2B

---

## Objective
The goal is to develop predictive models that can identify employees who are likely to leave the organization. By identifying high-risk attrition candidates, companies can proactively implement retention strategies.

---

## Dataset Link
The dataset used in this assignment is the **IBM HR Analytics Employee Attrition & Performance Dataset**.
- **Kaggle Link:** [IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

*Note: In accordance with the instructions, the dataset file `WA_Fn-UseC_-HR-Employee-Attrition.csv` is ignored in the git history and is not uploaded to GitHub.*

---

## Libraries Used
- **Pandas:** For data loading, manipulation, and exploratory data analysis.
- **NumPy:** For numerical operations and data handling.
- **Scikit-learn:** For data preprocessing, model building, and evaluation metrics.
- **Matplotlib & Seaborn:** For plotting confusion matrices and feature importance.

---

## Methodology

### 1. Data Understanding & Exploratory Analysis
- Loaded the 1,470 records containing 35 features.
- Identified the target variable as `Attrition` and split features into numerical and categorical categories.
- Generated summary statistics and structural info (`df.info()`).

### 2. Data Preprocessing
- Verified that there are no missing values in the dataset.
- Removed redundant features that provide no variance or predictive power:
  - `EmployeeCount` (always `1`)
  - `Over18` (always `'Y'`)
  - `StandardHours` (always `80`)
  - `EmployeeNumber` (unique ID)
- Encoded all categorical string features (e.g., `OverTime`, `BusinessTravel`, `MaritalStatus`) using Scikit-learn's `LabelEncoder`.
- Split the dataset into **80% Training** and **20% Testing** sets, utilizing stratification on `Attrition` to preserve class balance.

### 3. Model Development
Two classifiers were trained on the training dataset:
1. **Decision Tree Classifier:** Baseline model with default hyperparameters.
2. **Random Forest Classifier:** Ensemble model with 100 estimators.

---

## Results

### Model Performance Metrics
Evaluating both classifiers on the 20% test dataset yielded the following metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 78.23% | 31.91% | 31.91% | 31.91% |
| **Random Forest (100 estimators)** | 84.35% | 54.55% | 12.77% | 20.69% |

### Visualizations
The project generates two visualization plots:
1. `confusion_matrices.png`: Shows the confusion matrices for both models side-by-side.
2. `rf_feature_importance.png`: Displays the top 15 most important features for predicting attrition.

---

## Model Comparison

### Key Observations
1. **Accuracy:** The **Random Forest Classifier (84.35%)** outperforms the **Decision Tree Classifier (78.23%)** by over 6%. This indicates that the ensemble approach generalizes better to unseen test data.
2. **Precision vs. Recall:** The Decision Tree achieved a precision of **31.91%** and a recall of **31.91%**. In contrast, the Random Forest model scored a significantly higher precision of **54.55%** but a low recall of **12.77%**. This shows that when the Random Forest predicts an attrition (Yes), it is correct more than half the time, but it misses a large portion of actual attritions (high false negatives).
3. **F1-Score:** The F1-Score of the Decision Tree (**0.3191**) is higher than that of the Random Forest (**0.2069**) due to the Random Forest's low recall on the imbalanced class.
4. **Key Attrition Drivers:** According to the Feature Importance plot, the top features influencing attrition prediction in the Random Forest model are **MonthlyIncome**, **OverTime**, **Age**, **TotalWorkingYears**, and **DailyRate**. These highlight that financial compensation, work-life balance (overtime), and career maturity play critical roles.

---

## Conclusion
The Random Forest classifier performed better than the single Decision Tree in overall classification accuracy (84.35% vs. 78.23%) and precision (54.55% vs. 31.91%). Random Forest often outperforms Decision Trees because it is an ensemble model that aggregates predictions from multiple decision trees. This reduces the overall model variance, mitigates overfitting to training noise, and leads to smoother, more robust decision boundaries.

However, each model has notable limitations:
- **Decision Tree Limitation:** A single Decision Tree is highly sensitive to training data fluctuations and prone to overfitting, growing overly complex trees that fail to generalize well on test datasets.
- **Random Forest Limitation:** Random Forest models suffer when predicting highly imbalanced target classes. Because they optimize for global accuracy, they tend to heavily favor the majority class, resulting in low recall for the minority class (e.g., predicting attrition correctly for only 12.77% of actual attritions in this test set). They also lack the straightforward interpretability of a single decision tree.

---

## Bonus Challenge: Hyperparameter Tuning Experiment
We tuned the `max_depth` parameter of both the Decision Tree and the Random Forest models to analyze its effect on the test set.

### Decision Tree Tuning Results:
- **max_depth = 3:** Accuracy: 82.31% | Precision: 41.38% | Recall: 25.53% | F1-Score: 31.58%
- **max_depth = 5:** Accuracy: **84.35%** | Precision: 52.94% | Recall: 19.15% | F1-Score: 28.12%
- **max_depth = 7:** Accuracy: 81.97% | Precision: 41.67% | Recall: **31.91%** | F1-Score: **36.14%**
- **max_depth = 10:** Accuracy: 80.27% | Precision: 35.90% | Recall: 29.79% | F1-Score: 32.56%
- **max_depth = None (unlimited):** Accuracy: 78.23% | Precision: 31.91% | Recall: 31.91% | F1-Score: 31.91%

### Random Forest Tuning Results:
- **max_depth = 3:** Accuracy: 84.01% | Precision: 50.00% | Recall: 4.26% | F1-Score: 7.84%
- **max_depth = 5:** Accuracy: 83.67% | Precision: 44.44% | Recall: 8.51% | F1-Score: 14.29%
- **max_depth = 10:** Accuracy: 83.67% | Precision: 45.45% | Recall: 10.64% | F1-Score: 17.24%
- **max_depth = None (unlimited):** Accuracy: **84.35%** | Precision: **54.55%** | Recall: **12.77%** | F1-Score: **20.69%**

### Analysis:
Limiting the `max_depth` of the Decision Tree to **5** significantly improves generalization accuracy from **78.23%** (for unlimited depth) to **84.35%**. This is because deep trees overfit training noise, whereas limiting depth prunes redundant/noisy splits.
Conversely, Random Forest is less prone to overfitting deep trees because it aggregates predictions across multiple bootstrapped samples, so it achieves its highest test-set accuracy at `max_depth=None`.
