import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATASET_DIRECTORY = os.path.join(os.path.dirname(__file__), '../../src/data/dataset') 

def load_data():
    df_215 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-2-15-day.csv'))
    df_365 = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-headcount-ratio-at-3-65-day.csv'))
    df_post_tax = pd.read_csv(os.path.join(DATASET_DIRECTORY, 'cleaned_SDR-2025-poverty-rate-after-taxes-and-transfers.csv'))

    # merge on ISO and country
    df = df_215.merge(df_365, on=['ISO', 'Country'], suffixes=('_215', '_365'))
    df = df.merge(df_post_tax, on=['ISO', 'Country'], suffixes=('', '_post_tax'))

    return df

def prepare_features(df):
    # use 2000-2024 columns as features, 2025 as target
    feature_cols = []
    for suffix in ['_215', '_365']:
        for year in range(2000, 2025):  
            col = f"{year}{suffix}"
            if col in df.columns:
                feature_cols.append(col)
    target_col = '2025_215'  

    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_model():
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # save model
    joblib.dump(model, os.path.join(os.path.dirname(__file__), 'poverty_risk_model.pkl'))

if __name__ == "__main__":
    train_model()
