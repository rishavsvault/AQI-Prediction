import pandas as pd

df = pd.read_csv("data/Air_quality_data.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(["City", "Datetime"])
df["AQI"] = df["AQI"].interpolate()
df["Year"] = df["Datetime"].dt.year

from src.features import create_features
from src.models import linear_regression_model
from src.evaluate import evaluate_model

results = []

for city in df["City"].unique():

    city_df = df[df["City"] == city].copy()
    city_df = create_features(city_df)

    train = city_df[city_df["Year"].between(2021, 2023)]
    test = city_df[city_df["Year"] == 2024]

    columns_to_drop = ["AQI", "City", "Year", "Datetime", "AQI_Bucket"]

    X_train = train.drop(columns_to_drop, axis=1, errors="ignore")
    y_train = train["AQI"]

    X_test = test.drop(columns_to_drop, axis=1, errors="ignore")
    y_test = test["AQI"]

    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test.select_dtypes(include=["number"])

    print("\nChecking feature types for:", city)
    print(X_train.dtypes)

    model = linear_regression_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics.update({"City": city, "Phase": "Eval-2024"})

    results.append(metrics)

pd.DataFrame(results).to_csv("outputs/eval_phase1_linear.csv", index=False)

print("Phase 1 evaluation completed, saved to outputs/eval_phase1_linear.csv")

