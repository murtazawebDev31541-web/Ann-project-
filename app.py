import os
import json
import httpx
import numpy as np
import streamlit as st
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==========================================
# 1. FASTAPI BACKEND CONFIGURATION
# ==========================================
api_app = FastAPI(title="Keras Model Inference API")

# Global model container
model = None

@api_app.on_event("startup")
def load_model():
    global model
    config_path = "config.json"
    weights_path = "model.weights.h5"

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        return

    try:
        with open(config_path, "r") as f:
            model_config = f.read()

        model = keras.models.model_from_json(model_config)
        model.load_weights(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictRequest(BaseModel):
    features: list[float]

@api_app.post("/predict")
def predict(request: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    data = np.array(request.features)
    if data.shape[0] != 784:
        raise HTTPException(status_code=400, detail="Expected feature vector length 784.")

    input_data = np.expand_dims(data, axis=0)
    predictions = model.predict(input_data)
    predicted_class = int(np.argmax(predictions, axis=1)[0])
    probabilities = predictions[0].tolist()

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }

# Force FastAPI startup event on Streamlit cold-start
if model is None:
    load_model()

# Create an in-memory ASGI client to communicate with FastAPI without opening ports
client = httpx.Client(app=api_app, base_url="http://inprocess")

# ==========================================
# 2. STREAMLIT FRONTEND UI
# ==========================================
st.set_page_config(page_title="Model Inference Dashboard", layout="centered")

st.title("Sequential Neural Network Inference")
st.write("FastAPI backend & Streamlit UI running together on Streamlit Community Cloud.")

st.subheader("Input Features")
input_type = st.radio("Choose Input Method:", ("Generate Random Sample", "Manual Zero Array"))

if input_type == "Generate Random Sample":
    sample_input = np.random.rand(784).tolist()
else:
    sample_input = np.zeros(784).tolist()

if st.button("Run Prediction"):
    payload = {"features": sample_input}

    try:
        # Call FastAPI via direct ASGI memory protocol
        response = client.post("/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"**Predicted Class:** {result['predicted_class']}")
            st.subheader("Class Probabilities Distribution")
            st.bar_chart(result['probabilities'])
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Could not communicate with FastAPI app: {e}")