# 🏥 Depression Prediction Using Machine Learning
---

## 📌 Project Overview  
Mental health is a critical concern in today’s world. **Depression** affects millions of individuals, influencing their well-being, productivity, and quality of life. The ability to **predict depressive disorders** early can enable timely intervention and care.  

This project leverages **machine learning** and **data mining** techniques to classify individuals as **depressed (Y) or not depressed (N)** using responses from the **2020 Behavioral Risk Factor Surveillance System (BRFSS) Survey**, conducted by the **Centers for Disease Control and Prevention (CDC)**.  

The dataset comprises **5000 survey responses** with **276 attributes**, and various classification models are trained and evaluated to determine the most effective predictive model.  

---

## 🗂 Dataset  
The dataset used in this project is the **2020 Behavioral Risk Factor Surveillance System (BRFSS) Survey Data**, containing self-reported health-related risk behaviors, chronic health conditions, and preventive service usage.

- 📊 **Dataset Source**: CDC's BRFSS 2020  
- 📏 **Number of Attributes**: 276  
- 🧑‍🤝‍🧑 **Number of Survey Responses**: 5000  
- 🎯 **Target Variable**: `Class` (Depression: `Y` / No Depression: `N`)  

### 🔎 **Key Features**  
- **Demographic Information**: Age, Gender, Education, Income  
- **Health Conditions**: Chronic illnesses, BMI, Smoking habits  
- **Mental Health Indicators**: Sleep quality, Stress levels, Alcohol consumption  

---

## 🔬 Data Preprocessing  
### **1️⃣ Data Cleaning**  
- Removed redundant or irrelevant columns (survey metadata, technical codes, etc.).  
- Standardized date columns for consistency.  
- Converted categorical features to factors.  

### **2️⃣ Handling Missing Data**  
- Features with **>50% missing values** were **dropped**.  
- Other missing values were **imputed using probabilistic distribution** to maintain data integrity.  

### **3️⃣ Feature Selection Techniques**  
- **Lasso Regression**: Selected features based on absolute shrinkage and regularization.  
- **Information Gain**: Measured how informative each feature is for predicting depression.  
- **Principal Component Analysis (PCA)**: Reduced dimensionality while retaining variance.  

### **4️⃣ Data Balancing**  
Since the dataset had **imbalanced classes**, we applied **oversampling techniques**:  
- **Random Oversampling**: Duplicated minority class samples to balance the dataset.  
- **Synthetic Minority Over-sampling Technique (SMOTE)**: Created synthetic examples using nearest neighbors.  

---

## 🏗 Model Training & Evaluation  
### **🧠 Machine Learning Classifiers Used**  
We trained multiple classification models to compare performance:  
1. **XGBoost (Extreme Gradient Boosting)**  
2. **Gradient Boosting Machine (GBM)**  
3. **Logistic Regression**  
4. **Elastic Net Regularized Logistic Regression**  
5. **Linear Discriminant Analysis (LDA)**  
6. **Naïve Bayes Classifier**  

### **📊 Model Evaluation Metrics**  
Each model was evaluated using:  
✅ **Confusion Matrix** (True Positives, False Positives, True Negatives, False Negatives)  
✅ **Precision, Recall, F1-score**  
✅ **ROC Curve & AUC Score** (Assess trade-off between sensitivity and specificity)  
✅ **Matthews Correlation Coefficient (MCC)** (Measures quality of classification)  

---
## 📊 Results & Findings  
### **✔ Key Observations**  
- **XGBoost and GBM** consistently performed the best in terms of accuracy and AUC score.  
- **PCA & Information Gain** helped in selecting the most relevant features without losing prediction accuracy.  
- **Data Balancing using SMOTE** improved recall for the minority class (depressed individuals).  
- **Elastic Net Regularized Logistic Regression** provided a good balance between interpretability and performance.  

### **📌 Model Performance Comparison**  

| **Model** | **AUC Score (Higher is Better)** | **F1-Score** | **Accuracy** |
|-----------|----------------------------------|--------------|-------------|
| XGBoost | **0.78** | **0.74** | **77%** |
| GBM | **0.77** | **0.73** | **76%** |
| Logistic Regression | **0.75** | **0.71** | **74%** |
| Elastic Net | **0.74** | **0.70** | **73%** |
| LDA | **0.72** | **0.68** | **71%** |
| Naïve Bayes | **0.70** | **0.65** | **69%** |

---

## 📝 Conclusion  
1️⃣ **Feature selection techniques** helped **reduce dimensionality** without compromising model performance.  
2️⃣ **Balancing techniques** (Random Oversampling & SMOTE) **improved model recall** for predicting depression.  
3️⃣ **XGBoost & GBM outperformed** other models, making them the best choices for classification.  
4️⃣ **PCA & Information Gain** were effective in filtering important predictors.  

This study demonstrates that **machine learning** can be a valuable tool in identifying individuals at risk of depression, supporting early intervention strategies.

---

## 🎯 Future Work  
- Incorporate **deep learning models** (Neural Networks, Transformers) for better feature extraction.  
- **Use real-time data** from mental health surveys and online forums.  
- Deploy a **web-based dashboard** for mental health professionals to visualize insights.  

---
