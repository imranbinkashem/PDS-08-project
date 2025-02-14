
import pandas as pd

# Load the dataset
file_path = "/Users/mohammadimranbinkashem/programming_practice/PYTHON/iict-PDS-08/data_science/creditcard.csv"
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Check class distribution
class_counts = df["Class"].value_counts()
print("Class distribution:\n", class_counts)

# Separate fraud and non-fraud cases
df_fraud = df[df["Class"] == 1]
df_non_fraud = df[df["Class"] == 0].sample(n=len(df_fraud), random_state=42)

# Combine fraud and non-fraud samples to balance the dataset
df_balanced = pd.concat([df_fraud, df_non_fraud])

# Shuffle the dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
print("Balanced class distribution:\n", df_balanced["Class"].value_counts())

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define features and target variable
X = df_balanced.drop(columns=["Class"])
y = df_balanced["Class"]

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Check sizes of splits
print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for transaction amounts (fraud vs. non-fraud)
plt.figure(figsize=(6,4))
sns.boxplot(x=df_balanced["Class"], y=df_balanced["Amount"])
plt.title("Transaction Amounts for Fraud vs. Non-Fraud Cases")
plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
plt.ylabel("Transaction Amount")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_balanced.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

# Train Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Train SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate Naïve Bayes
report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
roc_nb = roc_auc_score(y_test, nb_model.predict_proba(X_test)[:,1])

# Evaluate SVM
report_svm = classification_report(y_test, y_pred_svm, output_dict=True)
roc_svm = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:,1])

# Display results
print("Naïve Bayes Metrics:")
print(report_nb)
print("Naïve Bayes AUC-ROC Score:", roc_nb)

print("\nSVM Metrics:")
print(report_svm)
print("SVM AUC-ROC Score:", roc_svm)

# Formatting the evaluation results for better readability
import pandas as pd

# Create DataFrames for classification reports
report_nb_df = pd.DataFrame(report_nb).transpose()
report_svm_df = pd.DataFrame(report_svm).transpose()

# Display results in a structured format
formatted_results = {
    "Model": ["Naïve Bayes", "SVM"],
    "Precision (Fraud)": [report_nb["1"]["precision"], report_svm["1"]["precision"]],
    "Recall (Fraud)": [report_nb["1"]["recall"], report_svm["1"]["recall"]],
    "F1-score (Fraud)": [report_nb["1"]["f1-score"], report_svm["1"]["f1-score"]],
    "Accuracy": [report_nb["accuracy"], report_svm["accuracy"]],
    "AUC-ROC Score": [roc_nb, roc_svm]
}

# Convert to DataFrame
results_df = pd.DataFrame(formatted_results)
print(results_df)
# Display results in a structured format
print(111111111111111111)
