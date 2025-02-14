
<html lang="en">

<body>
    <h1>Detecting Credit Card Fraud Using Naive Bayes</h1> 
    <h2>Overview</h2>
    <p>This project aims to detect fraudulent credit card transactions using machine learning techniques, specifically comparing <b><i>Naïve Bayes and Support Vector Machine (SVM)</i></b> models.</p>
    <h2>Dataset</h2>
    <p><strong>Source:</strong> <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud" target="_blank">Credit Card Fraud Dataset</a></p>
    <p>The dataset contains anonymized transaction details with labeled fraud and non-fraud cases.</p>
    
   <h2>Objectives</h2>
    <ul>
        <li>Detect fraudulent transactions effectively.</li>
        <li>Compare <b><i>Naïve Bayes with SVM</i></b> for fraud detection.</li>
        <li>Evaluate model performance using precision-recall metrics.</li>
    </ul>
    
  <h2>Preprocessing</h2>
    <ul>
        <li>Feature scaling applied to the <code>Amount</code> column using <strong>StandardScaler</strong>.</li>
        <li>Dataset split into <strong>training (80%)</strong> and <strong>testing (20%)</strong> sets.</li>
    </ul>
        <h2>Visualization</h2>
    <ul>
        <li><strong>Transaction Amount Distribution:</strong> Histograms to compare fraud vs. non-fraud transactions.</li>
        <li><strong>Precision-Recall Curve:</strong> To assess model performance.</li>
    </ul>
    
  <h2>Modeling</h2>
    <ul>
        <li><strong>Naïve Bayes Model</strong>
            <ul>
                <li>Trained on the preprocessed dataset.</li>
                <li>Predicts fraudulent transactions.</li>
            </ul>
        </li>
        <li><strong>SVM Model</strong>
            <ul>
                <li>Trained for comparison.</li>
                <li>Evaluated based on precision-recall metrics.</li>
            </ul>
        </li>
    </ul>
    
  <h2>Evaluation</h2>
    <ul>
        <li><strong>Classification Report:</strong> Precision, recall, and F1-score.</li>
        <li><strong>ROC-AUC Score:</strong> To compare model performance.</li>
    </ul>
        <h2>How to Run</h2>
    <ol>
        <li>Install dependencies: <code>pip install pandas numpy seaborn scikit-learn matplotlib</code></li>
        <li>Download dataset from Kaggle and place it in the project directory.</li>
        <li>Run <code>fraud_detection.py</code> to preprocess data, train models, and generate evaluation metrics.</li>
    </ol>
    
  <h2>Contributing</h2>
    <p>Feel free to fork this repository and enhance the model performance or add new fraud detection techniques.</p>
    
  
</body>
</html>
