# ✈️ Flight Ticket Price Prediction

This project focuses on predicting the prices of flight tickets using machine learning. It uses two regression models – **Linear Regression** and **Random Forest Regressor** – to predict ticket prices based on features such as airline, source, destination, duration, stops, and timings.

---

## 📌 Problem Statement

Build a machine learning model that accurately predicts flight ticket prices based on flight-related features. This can help travelers estimate costs and assist platforms in dynamic pricing strategies.

---

## 📂 Dataset

The dataset contains information about:
- `Airline` – Airline carrier  
- `Source` – Departure city  
- `Destination` – Arrival city  
- `Date_of_Journey`, `Dep_Time`, `Arrival_Time` – Date and time of flight  
- `Duration` – Total travel time  
- `Total_Stops` – Number of stops  
- `Price` – (Target) Ticket price in INR

📥 Dataset Source:  
🔗 [Flight Price Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

---

## 🛠️ Technologies Used

- **Python**
- **pandas**, **NumPy** – Data preprocessing
- **matplotlib**, **seaborn** – Visualization
- **scikit-learn** – ML model training and evaluation
- **Linear Regression**, **Random Forest Regressor**

---

## 📊 EDA & Feature Engineering

- Extracted new features from `Date_of_Journey`, `Dep_Time`, `Arrival_Time`
- Converted `Duration` to total minutes
- One-hot encoded categorical variables: `Airline`, `Source`, `Destination`, etc.
- Removed irrelevant or duplicate features
- Split dataset into train and test using `train_test_split`

---

## 🧠 Model Training

### 🔹 Linear Regression

```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
