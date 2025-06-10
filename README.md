# Classification_Models_on_Loan_Approval_Dataset

A study on how different classification models performed on a loan prediction dataset containing applicant financial and demographic data to determine loan approval outcomes.

# Objective

The objective of this repository is to compare the performance of various popular classification models on a loan approval dataset. This dataset includes features such as age, education, income, employment experience, credit history, and loan details. The goal is to evaluate how well different models can predict whether a loan will be approved or rejected, using key performance metrics: Accuracy, Precision, Recall, and F1-Score. This comparison helps in identifying suitable models for real-world financial decision-making applications.

# Dataset

The dataset used in this project is a structured tabular dataset containing information about loan applicants. Each row represents an applicant and includes various numerical and categorical features along with the target label: whether the loan was approved (1) or rejected (0).

- Features include: age, education, income, employment experience, credit score, loan amount, interest rate, and categorical variables like loan intent and home ownership status.
- Target: `loan_status` (1 for approved, 0 for rejected)

# Classification Models Overview

**1. Logistic Regression**

A linear model suitable for binary classification tasks. It is interpretable and serves as a strong baseline for predicting loan approval outcomes.

**2. Kernel Support Vector Machine (Kernel SVM)**

A powerful classifier that uses kernel functions to handle non-linearly separable data. Offers high accuracy but can be computationally intensive.

**3. K-Nearest Neighbors (KNN)**

A simple, instance-based learning algorithm that classifies data points based on the majority class of their nearest neighbors. Performance improves with proper scaling and tuning.

**4. Naive Bayes**

A probabilistic model based on Bayes’ Theorem. While simple and fast, it assumes feature independence, which may limit performance on correlated financial data.

**5. Decision Tree**

Builds interpretable tree structures for classification. Captures non-linear patterns but can overfit without proper pruning or regularization.

**6. Random Forest**

An ensemble of decision trees that enhances accuracy and reduces overfitting. Performs robustly across most classification tasks, including loan prediction.

# Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.89     | 0.77      | 0.74   | 0.76     |
| Kernel SVM          | 0.91     | 0.85      | 0.75   | 0.80     |
| KNN                 | 0.90     | 0.84      | 0.69   | 0.76     |
| Naive Bayes         | 0.74     | 0.46      | 1.00   | 0.63     |
| Decision Tree       | 0.91     | 0.86      | 0.72   | 0.78     |
| Random Forest       | 0.92     | 0.90      | 0.73   | 0.81     |

*Note: Replace TBD with actual values after running the models.*

# Key Insights

Logistic Regression:
A strong baseline model with 89% accuracy and balanced performance across all metrics. Its simplicity and interpretability make it a reliable choice for binary financial classification, especially when model transparency is important.

Kernel SVM:
Delivered 91% accuracy with the second-highest F1-score (0.80). It handled the feature complexity well and offered a strong balance of precision and generalization, though it's more computationally intensive.

K-Nearest Neighbors (KNN):
Achieved 90% accuracy, with strong precision (0.84) but lower recall (0.69), indicating it's conservative in predicting approvals. Best used when interpretability isn’t a priority and computational resources allow.

Naive Bayes:
Showed the lowest accuracy (74%), but perfect recall (1.00), meaning it predicted all approved loans correctly—but at the cost of many false positives (low precision). It may be suitable in cases where minimizing false negatives is critical, but not when precision is required.

Decision Tree:
Offered 91% accuracy, with good precision (0.86) and moderate recall (0.72). Its interpretability and fast training make it suitable for explaining decisions, though it may require tuning to avoid overfitting.

Random Forest:
The best overall performer, with 92% accuracy, highest precision (0.90), and highest F1-score (0.81). It combines robustness with high performance, making it ideal for production use where both accuracy and stability are crucial.
# Conclusion

This project highlights how various classification models perform on a real-world loan approval dataset. While **Random Forest** and **Kernel SVM** are expected to offer the best predictive power, **Logistic Regression** remains a strong baseline with solid interpretability. The choice of model should consider accuracy, computational efficiency, and interpretability depending on the deployment context.

