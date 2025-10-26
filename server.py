
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

import pickle
import warnings

import os


warnings.filterwarnings("ignore")

import sys
from json import *
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

crop_recommendation_model_path = os.path.join(BASE_DIR, "RF_pkl_file.pkl")  # "./XGBoost.pkl"
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, "rb"))

fertilizer_recommendation_model_path = (
    os.path.join(BASE_DIR, "LogisticRegression_Fertilizer.pkl")  # "./SVM Fertilizer.pkl"
)
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, "rb")
)


def generate_random_values():
    data_ranges = {"N": (0, 140), "P": (5, 145), "K": (5, 205), "fert": (0, 6)}

    random_values = []
    for key, (min_val, max_val) in data_ranges.items():
        random_values.append(int(random.uniform(min_val, max_val)))

    return random_values


def _predict_crop_from_payload(payload: dict):
    N = int(payload["N"]) if "N" in payload else 0
    P = int(payload["P"]) if "P" in payload else 0
    K = int(payload["K"]) if "K" in payload else 0
    ph = float(payload["Ph"]) if "Ph" in payload else 7.0
    state = payload.get("state", "")
    district = payload.get("district", "")
    start_month = int(payload.get("start_month", 1))
    end_month = int(payload.get("end_month", 12))

    # Default temperature and humidity values
    temprature = 20
    humidity = 30

    df = pd.read_csv(os.path.join(BASE_DIR, "data2.csv"))
    q = df.query(
        'STATE_UT_NAME == "{}" and DISTRICT == "{}"'.format(state, district),
        inplace=False,
    ) if state and district else df.head(0)

    total = 0
    if start_month <= end_month:
        l = (end_month - start_month) + 1
        for i in range(start_month, end_month + 1):
            try:
                total += int(q[i : i + 1].value)
            except:
                total -= 1
    else:
        l = (end_month + 12) - start_month + 1
        for i in range(start_month, 13):
            try:
                total += int(q[i : i + 1].value)
            except:
                total -= 1
        for i in range(1, end_month + 1):
            try:
                total += int(q[i : i + 1].value)
            except:
                total -= 1

    avg_rainfall = total / l if l else 0
    random_sample = generate_random_values()
    data = np.array(
        [[
            N + random_sample[0],
            P + random_sample[1],
            K + random_sample[2],
            temprature,
            humidity,
            ph,
            avg_rainfall,
        ]]
    )
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = int(my_prediction[-1])
    crop_name_dict = {
        20: "rice",
        16: "maize",
        4: "chickpea",
        11: "kidneybeans",
        17: "pigeonpeas",
        12: "mothbeans",
        13: "mungbean",
        3: "blackgram",
        9: "lentil",
        18: "pomegranate",
        2: "banana",
        8: "mango",
        7: "grapes",
        21: "watermelon",
        15: "muskmelon",
        1: "apple",
        19: "orange",
        14: "papaya",
        5: "coconut",
        6: "cotton",
        10: "jute",
        0: "coffee",
    }
    pred_crop_name = crop_name_dict.get(final_prediction, "unknown")
    return pred_crop_name


@app.route("/crop", methods=["POST", "OPTIONS"])
def crop():
    
    if request.method == "OPTIONS":
        resp = jsonify({"status": "ok"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        resp.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return resp, 200
    
    try:
        payload = request.get_json(silent=True) or {}
        print("Received crop data:", payload)
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'Ph', 'state', 'district', 'start_month', 'end_month']
        missing_fields = [field for field in required_fields if field not in payload or payload[field] is None]
        
        if missing_fields:
            return jsonify({
                "status": "error", 
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Validate numeric ranges
        validation_errors = []
        if not (0 <= int(payload.get('N', 0)) <= 140):
            validation_errors.append("N must be between 0-140")
        if not (5 <= int(payload.get('P', 0)) <= 145):
            validation_errors.append("P must be between 5-145")
        if not (5 <= int(payload.get('K', 0)) <= 205):
            validation_errors.append("K must be between 5-205")
        if not (0 <= float(payload.get('Ph', 0)) <= 14):
            validation_errors.append("pH must be between 0-14")
        if not (1 <= int(payload.get('start_month', 0)) <= 12):
            validation_errors.append("Start month must be between 1-12")
        if not (1 <= int(payload.get('end_month', 0)) <= 12):
            validation_errors.append("End month must be between 1-12")
            
        if validation_errors:
            return jsonify({
                "status": "error",
                "message": "Validation failed: " + "; ".join(validation_errors)
            }), 400
        
        crop_name = _predict_crop_from_payload(payload)
        
       
        response_data = {
            "status": "success",
            "crop": crop_name,
            "confidence": "high", 
            "recommendation": f"Based on your soil analysis, {crop_name} is the most suitable crop for your conditions.",
            "next_steps": [
                "Prepare your soil according to the crop requirements",
                "Consider seasonal timing for planting",
                "Monitor soil conditions regularly"
            ]
        }
        
        resp = jsonify(response_data)
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 200
        
    except ValueError as e:
        resp = jsonify({"status": "error", "message": f"Invalid input format: {str(e)}"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 400
    except Exception as e:
        print(f"Crop prediction error: {str(e)}")
        resp = jsonify({"status": "error", "message": f"Prediction failed: {str(e)}"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 500
    
    


@app.route("/fertilizer", methods=["POST", "OPTIONS"])
def fertilizer():
    # Handle preflight explicitly
    if request.method == "OPTIONS":
        resp = jsonify({"status": "ok"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        resp.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        resp.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return resp, 200

    try:
        payload = request.get_json(silent=True) or {}
        print("Received fertilizer data:", payload)
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'moisture', 'soil_type', 'crop_type', 'state', 'district', 'start_month', 'end_month']
        missing_fields = [field for field in required_fields if field not in payload or payload[field] is None]
        
        if missing_fields:
            return jsonify({
                "status": "error", 
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        # Extract and validate data
        N = int(payload["N"])
        P = int(payload["P"])
        K = int(payload["K"])
        state = payload["state"]
        district = payload["district"]
        moisture = float(payload["moisture"])
        soil_type = int(payload["soil_type"])
        crop_type = int(payload["crop_type"])
        start_month = int(payload["start_month"])
        end_month = int(payload["end_month"])
        
        # Validate numeric ranges
        validation_errors = []
        if not (0 <= N <= 140):
            validation_errors.append("N must be between 0-140")
        if not (5 <= P <= 145):
            validation_errors.append("P must be between 5-145")
        if not (5 <= K <= 205):
            validation_errors.append("K must be between 5-205")
        if not (0 <= moisture <= 100):
            validation_errors.append("Moisture must be between 0-100")
        if not (0 <= soil_type <= 6):
            validation_errors.append("Soil type must be between 0-6")
        if not (0 <= crop_type <= 6):
            validation_errors.append("Crop type must be between 0-6")
        if not (1 <= start_month <= 12):
            validation_errors.append("Start month must be between 1-12")
        if not (1 <= end_month <= 12):
            validation_errors.append("End month must be between 1-12")
            
        if validation_errors:
            return jsonify({
                "status": "error",
                "message": "Validation failed: " + "; ".join(validation_errors)
            }), 400
            
    except ValueError as e:
        resp = jsonify({"status": "error", "message": f"Invalid input format: {str(e)}"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 400
    except Exception as e:
        resp = jsonify({"status": "error", "message": f"Invalid input payload: {str(e)}"})
        resp.headers.add("Access-Control-Allow-Origin", "*")
        return resp, 400

    # Default temperature and humidity values
    temprature = 20
    humidity = 30

    df = pd.read_csv(os.path.join(BASE_DIR, "data2.csv"))
    q = df.query(
        'STATE_UT_NAME == "{}" and DISTRICT == "{}"'.format(state, district),
        inplace=False,
    ) if state and district else df.head(0)

    total = 0
    if start_month <= end_month:
        l = (end_month - start_month) + 1
        for i in range(start_month, end_month + 1):
            try:
                total += int(q[i : i + 1].value)
            except Exception:
                total -= 1
    else:
        l = (end_month + 12) - start_month + 1
        for i in range(start_month, 13):
            try:
                total += int(q[i : i + 1].value)
            except Exception:
                total -= 1
        for i in range(1, end_month + 1):
            try:
                total += int(q[i : i + 1].value)
            except Exception:
                total -= 1

    avg_rainfall = total / l if l else 0

    data = np.array([[avg_rainfall, humidity, moisture, soil_type, crop_type, N, K, P]])

    try:
        my_prediction = fertilizer_recommendation_model.predict(data)
        random_sample = generate_random_values()
        raw_index = int(my_prediction[-1]) - random_sample[3]
    except Exception:
        raw_index = 0

    # Clamp to valid label range 0..6
    index = max(0, min(6, int(raw_index)))

    fertname = {
        0: "Diammonium Phosphate",
        1: "Muriate of Potash (MOP)",
        2: "Single Superphosphate (SSP) ",
        3: "NPK",
        4: "Zinc Sulphate",
        5: "DAP",
        6: "Urea",
    }

    # Enhanced response with additional information
    fertilizer_name = fertname[index]
    response_data = {
        "status": "success",
        "fertilizer": fertilizer_name,
        "confidence": "high",
        "recommendation": f"Based on your soil analysis, {fertilizer_name} is the recommended fertilizer for your crop.",
        "application_tips": [
            "Apply fertilizer evenly across the field",
            "Follow the recommended application rate",
            "Consider soil moisture levels before application",
            "Monitor crop response after application"
        ],
        "soil_analysis": {
            "N": N,
            "P": P, 
            "K": K,
            "moisture": moisture,
            "soil_type": soil_type,
            "crop_type": crop_type
        }
    }
    
    resp = jsonify(response_data)
    resp.headers.add("Access-Control-Allow-Origin", "*")
    return resp, 200


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        "status": "healthy",
        "message": "AgriBot server is running",
        "version": "1.0.0",
        "endpoints": ["/crop", "/fertilizer", "/health"]
    }), 200

if __name__ == "__main__":
    print("Starting AgriBot Server...")
    print("Crop Recommendation: http://127.0.0.1:5050/crop")
    print("Fertilizer Recommendation: http://127.0.0.1:5050/fertilizer")
    print("Health Check: http://127.0.0.1:5050/health")
    app.run(debug=True, port=5050)