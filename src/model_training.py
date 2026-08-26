import polars as pl
import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def train_momentum_model():
    print("Loading high-frequency dataset...")
    try:
        df = pl.read_parquet("xgboost_training_data.parquet")
    except FileNotFoundError:
        print("Dataset missing. Run options_pipeline.py first.")
        return
    
    # 1. Enforce strict chronological sorting to prevent look-ahead bias
    df = df.sort("timestamp_1m")
    
    # 2. Isolate Features (X) and Target (y)
    features = ["opt_put_call_ratio", "opt_call_shock"]
    X = df.select(features).to_numpy()
    y = df.select("Realized_Target_EV").to_numpy().ravel()
    
    print("\n--- Starting Walk-Forward Optimization ---")
    
    # 3. 5-Fold TimeSeriesSplit
    # This ensures the model only ever trains on the past to predict the future.
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    out_of_sample_maes = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Instantiate the regressor. Keeping max_depth shallow (3) prevents over-fitting to noise.
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            objective="reg:squarederror"
        )
        
        # Train on the expanding window
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate on the unseen "future" fold
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        out_of_sample_maes.append(mae)
        
        print(f"Fold {fold} | Out-of-Sample MAE: {mae * 100:.3f}%")
        fold += 1
        
    print("\n==========================================")
    print("CROSS-VALIDATION COMPLETE")
    print("==========================================")
    print(f"Average Out-of-Sample Error: {np.mean(out_of_sample_maes) * 100:.3f}%")
    
    # 4. Train the final production model on 100% of the dataset
    print("\nTraining Final Production Model...")
    final_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        objective="reg:squarederror"
    )
    final_model.fit(X, y)
    
    # 5. Save the model in the institutional standard JSON format
    model_filename = "options_flow_model_v1.json"
    final_model.save_model(model_filename)
    print(f"Success. Production engine saved as '{model_filename}'.")

if __name__ == "__main__":
    train_momentum_model()