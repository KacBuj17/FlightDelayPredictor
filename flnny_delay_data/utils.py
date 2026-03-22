from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

def get_preprocessor(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    return preprocessor

def train_nn_model(model, X_train, y_train, X_val=None, y_val=None,
                   epochs=10, batch_size=64, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if isinstance(X_train, torch.Tensor):
        X_train_tensor = X_train.detach().clone().to(device)
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)

    if isinstance(y_train, torch.Tensor):
        y_train_tensor = y_train.detach().clone().to(device)
    else:
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    if X_val is not None and y_val is not None:
        if isinstance(X_val, torch.Tensor):
            X_val_tensor = X_val.detach().clone().to(device)
        else:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
        if isinstance(y_val, torch.Tensor):
            y_val_tensor = y_val.detach().clone().to(device)
        else:
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            yb = yb.view_as(outputs)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(loader.dataset)

        val_loss = None
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                y_val_tensor_view = y_val_tensor.view_as(val_outputs)
                val_loss = criterion(val_outputs, y_val_tensor_view).item()

        print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}",
              f"{', Val Loss: %.4f' % val_loss if val_loss is not None else ''}")

    return model


def evaluate_model(model, X_test, y_test):
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred = model(X_test_tensor).squeeze().cpu().detach().numpy()
    else:
        y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse


def time_to_minutes_col(col):
    col = pd.to_datetime(col, errors='coerce')
    return col.dt.hour * 60 + col.dt.minute
