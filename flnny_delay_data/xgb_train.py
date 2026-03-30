import pandas as pd
from sklearn.model_selection import train_test_split
from models import build_xgb_model
from utils import evaluate_model, time_to_minutes_col
import json
import joblib

def main():
    df = pd.read_csv("../resources/data/flight_with_weather_2016.csv")
    df.dropna(inplace=True)

    df['CRS_DEP_MIN'] = time_to_minutes_col(df['CRS_DEP_TIME'])
    df['DEP_MIN'] = time_to_minutes_col(df['DEP_TIME'])
    df['CRS_ARR_MIN'] = time_to_minutes_col(df['CRS_ARR_TIME'])
    df['ARR_MIN'] = time_to_minutes_col(df['ARR_TIME'])

    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], errors='coerce')
    df['FL_MONTH'] = df['FL_DATE'].dt.month
    df['FL_DAY'] = df['FL_DATE'].dt.day
    df['FL_WEEKDAY'] = df['FL_DATE'].dt.weekday

    numerical_features = [
        'CRS_DEP_MIN', 'DEP_MIN', 'CRS_ARR_MIN', 'ARR_MIN',
        'TAXI_OUT', 'TAXI_IN', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME',
        'AIR_TIME', 'O_TEMP', 'O_PRCP', 'O_WSPD',
        'D_TEMP', 'D_PRCP', 'D_WSPD', 'FL_MONTH', 'FL_DAY', 'FL_WEEKDAY'
    ]
    categorical_features = []
    target = 'DEP_DELAY'

    X = df[numerical_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = build_xgb_model(numerical_features, categorical_features)
    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "../resources/models/xgb_model.pkl")

    mae, mse, rmse = evaluate_model(xgb_model, X_test, y_test)
    xgb_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}

    with open("../resources/stats/xgb/xgb_metrics.json", "w") as f:
        json.dump(xgb_metrics, f)

    print("XGBoost:", xgb_metrics)


if __name__ == "__main__":
    main()