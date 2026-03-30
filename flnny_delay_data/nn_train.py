import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from models import FullyConnectedNN
from utils import train_nn_model, evaluate_model, time_to_minutes_col, get_preprocessor
import json
import os


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

    preprocessor = get_preprocessor(numerical_features, categorical_features)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    X_train_nn = preprocessor.fit_transform(X_train)
    X_val_nn = preprocessor.transform(X_val)
    X_test_nn = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_nn, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    nn_model = FullyConnectedNN(input_dim=X_train_tensor.shape[1])
    nn_model = train_nn_model(
        nn_model,
        X_train_tensor,
        y_train_tensor,
        X_val=X_val_tensor,
        y_val=y_val_tensor,
        epochs=5,
        batch_size=64,
        lr=0.001
    )

    os.makedirs("../resources/models", exist_ok=True)
    torch.save(nn_model.state_dict(), "../resources/models/nn_model.pth")

    mae, mse, rmse = evaluate_model(nn_model, X_test_tensor, y_test_tensor)
    nn_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse}

    with open("../resources/stats/fc_linear_nn/nn_metrics.json", "w") as f:
        json.dump(nn_metrics, f)

    print(f"NN Test MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
