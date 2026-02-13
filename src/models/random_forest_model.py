from sklearn.ensemble import RandomForestRegressor

def random_forest_model():
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    return model
