# 🔍 Customer Churn Prediction using Machine Learning

Welcome to the Customer Churn Prediction Project! 🧠📊  
This project helps telecom companies **predict whether a customer is likely to leave (churn)** or stay — so they can take proactive actions.

---

## 🚀 Project Overview

This project uses various machine learning algorithms like:
- 🌳 Decision Tree
- 🌲 Random Forest
- ⚡ XGBoost

To **predict customer churn** based on features like contract type, monthly charges, internet service, etc.

---

## 🧰 Tools & Libraries Used

| Tool/Library           | Purpose                          |
|------------------------|----------------------------------|
| `pandas`, `numpy`      | Data loading & manipulation      |
| `matplotlib`, `seaborn`| Data visualization               |
| `scikit-learn`         | ML models, metrics & splitting   |
| `xgboost`              | Boosted tree model               |
| `imblearn.SMOTE`       | Handle class imbalance           |
| `pickle`               | Save and load the trained model  |

---

## 📊 Workflow & Steps

### 1️⃣ Load the Dataset
Customer data is loaded using pandas and inspected.

### 2️⃣ Data Cleaning
- Dropped unnecessary columns like `customerID`.
- Converted data types.
- Handled missing values.

### 3️⃣ Feature Encoding
Categorical columns were converted into numerical format using `LabelEncoder`.

### 4️⃣ Train-Test Split
Data is split into:
- 🧪 **80% training**
- 🧾 **20% testing**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
### 5️⃣ Handle Class Imbalance with SMOTE
The dataset was imbalanced, meaning fewer customers had churned compared to those who hadn’t.  
To fix this, we used **SMOTE** (Synthetic Minority Over-sampling Technique) to create synthetic samples of the **minority class**.


### 6️⃣ Train Models

We trained and evaluated three machine learning models:

🌳 Decision Tree

🌲 Random Forest

⚡ XGBoost
```
model.fit(X_train_smote, y_train_smote)
```
### 7️⃣ Model Evaluation
Used metrics like:

✅ Accuracy

### 📉 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

### 📊 Model Accuracy Comparison

| Model           | Accuracy |
|----------------|----------|
| Decision Tree  | 82%      |
| Random Forest  | 84% ✅   |
| XGBoost        | 91% 🔥   |


### 8️⃣ Save the Best Model
The trained Random Forest model was saved using pickle.

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
### 9️⃣ Predict New Data
Load the model anytime and predict churn for new customer input.
![prediction](prediction_output.png)


### 🔮 Real-World Impact
Using this model, companies can:

🚫 Reduce customer loss

💸 Save revenue

🎯 Target at-risk customers

📁 Project Structure

📦 Customer-Churn-Prediction/
├── 📄 README.md
├── 📊 dataset.csv
├── 📓 notebook.ipynb
├── 🧠 customer_churn_model.pkl

🙌 Let's Connect!
Feel free to ⭐ this repo if you found it helpful!
For feedback or collaboration:
📧 riznxz@example.com

