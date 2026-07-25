from fastapi import FastAPI, Query
import joblib
import numpy as np

app = FastAPI(title="K-Means Cluster Prediction Engine")

# Load saved artifacts
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

@app.get("/")
def home():
    return {"message": "K-Means Engine is running."}

@app.get("/predict")
def predict_cluster(
    income: float = Query(
        ..., 
        ge=0.0, # validation: income must be non-negative
        description="Annual Income in $k (must be >= 0)" # validation: income must be non-negative
    ), 
    score: float = Query(
        ..., 
        ge=1.0, # validation: score minimum value 1.0
        le=100.0, # validation: score maximum value 100.0
        description="Spending Score (must be between 1 and 100)"
    )
):
    # 1. Format input into 2D numpy array
    new_data = np.array([[income, score]])
    
    # 2. Scale features using saved scaler
    new_data_scaled = scaler.transform(new_data)
    
    # 3. Predict cluster
    cluster_label = int(kmeans.predict(new_data_scaled)[0])
    
    return {
        "inputs": {"Income": income, "Score": score},
        "predicted_cluster": cluster_label
    }
