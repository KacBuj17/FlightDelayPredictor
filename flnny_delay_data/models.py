from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import torch.nn as nn

from flnny_delay_data.utils import get_preprocessor


def build_linear_model(numerical_features, categorical_features):
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    return model


def build_xgb_model(numerical_features, categorical_features, n_estimators=200, max_depth=5, learning_rate=0.1):
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   learning_rate=learning_rate, random_state=42))
    ])
    return model


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64, 32]):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.out = nn.Linear(hidden_units[2], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x
