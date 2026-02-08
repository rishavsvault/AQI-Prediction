from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np, pandas as pd

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "MAE": round(mean_absolute_error(y_test, preds), 2)
    }
