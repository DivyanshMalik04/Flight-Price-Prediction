# ✈️ Flight Price Prediction using Machine Learning

This project predicts flight ticket prices using supervised machine learning techniques. The model is trained on a Kaggle dataset containing features like airline, source, destination, duration, and more.

## 📁 Dataset

- Source: [Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- Columns include Airline, Source, Destination, Duration, Total Stops, etc.

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## 🧠 Models Used

- Linear Regression
- Random Forest Regressor

## 📊 Evaluation Metrics

| Model                 | MAE      | RMSE     | R² Score |
|----------------------|----------|----------|----------|
| Linear Regression     | 2036.82  | 3043.03  | 0.56     |
| Random Forest Regressor | 1201.02 | 2165.11  | 0.77     |

## 🔧 Feature Engineering

- Encoded categorical features (Airline, Source, Destination)
- Transformed Duration column
- Handled missing values

## 🚀 Future Scope

- Deploy as a responsive web app using Flask/Streamlit
- Add real-time inputs for flight predictions
- Improve model performance through hyperparameter tuning

## 💻 Getting Started

```bash
git clone https://github.com/DivyanshMalik04/Flight-Price-Prediction.git
cd Flight-Price-Prediction
# Open the notebook and run cells in order
