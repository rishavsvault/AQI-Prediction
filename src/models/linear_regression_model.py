from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def linear_regression_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
