import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score

# Load dataset
df = pd.read_csv('creditcard.csv')

# Check for missing values
print("Missing values:")
print(df.isnull().sum().sum())

# Class distribution
print("Class distribution:")
print(df['Class'].value_counts())

# Feature scaling
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Splitting dataset
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Visualization
plt.figure(figsize=(10,5))
sns.histplot(data=df[df['Class'] == 0], x='Amount', bins=50, color='blue', label='Non-Fraud', kde=True)
sns.histplot(data=df[df['Class'] == 1], x='Amount', bins=50, color='red', label='Fraud', kde=True)
plt.legend()
plt.title("Transaction Amount Distribution for Fraud vs Non-Fraud Cases")
plt.show()

# Train Naïve Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Train SVM Model
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Evaluation
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, nb_preds))
print("SVM Classification Report:")
print(classification_report(y_test, svm_preds))

# Precision-Recall Curve
nb_probs = nb_model.predict_proba(X_test)[:, 1]
svm_probs = svm_model.predict_proba(X_test)[:, 1]
precision_nb, recall_nb, _ = precision_recall_curve(y_test, nb_probs)
precision_svm, recall_svm, _ = precision_recall_curve(y_test, svm_probs)

plt.figure(figsize=(8,6))
plt.plot(recall_nb, precision_nb, label='Naïve Bayes', linestyle='--')
plt.plot(recall_svm, precision_svm, label='SVM', linestyle='-')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# ROC-AUC Score
print("Naïve Bayes AUC Score:", roc_auc_score(y_test, nb_probs))
print("SVM AUC Score:", roc_auc_score(y_test, svm_probs))
