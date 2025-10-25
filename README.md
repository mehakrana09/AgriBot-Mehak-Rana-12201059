# 🌾 AgriBot - Smart Agriculture Assistant

BY MEHAK RANA

AgriBot is an intelligent agriculture recommendation system that helps farmers make informed decisions about crop selection and fertilizer application based on soil conditions, weather data, and location information.

## Technical Details

### Frontend Technologies
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **JavaScript**: Interactive functionality


### Backend Technologies
- **Flask**: Web framework
- **Scikit-learn**: Machine learning models
- **Pandas**: Data processing
- **NumPy**: Numerical computations
- **Requests**: API integrations

### Machine Learning Models
- **Random Forest**: Crop recommendation
- **Logistic Regression**: Fertilizer recommendation
- **Regional Data Integration**: Location-based climate analysis


## Dataset 
- **Kaggle** : Dataset is taken from the kaggle 
Fertlizer Dataset: [fertilizer](https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction)
Crop Dataset: [Crop](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)




## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
python start.py
```

### Option 2: Manual Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python server.py
   ```

3. **Open the Frontend**:
   - Open `index.html` in your web browser
   - Or serve it using a local server

## Usage Guide

### Crop Recommendation
1. Click on "Crop Recommendation" mode
2. Provide the following information:
   - **Nitrogen (N)**: 0-140
   - **Phosphorus (P)**: 5-145  
   - **Potassium (K)**: 5-205
   - **Soil pH**: 0-14
   - **State & District**: Your location
   - **Growing Season**: Start and end months (1-12)

### Fertilizer Recommendation
1. Click on "Fertilizer Recommendation" mode
2. Provide the following information:
   - **Soil Nutrients**: N, P, K values
   - **Soil Moisture**: 0-100%
   - **Soil Type**: 0-6 (Sandy, Loamy, Clayey, etc.)
   - **Crop Type**: 0-6 (Rice, Maize, Wheat, etc.)
   - **Location**: State and district
   - **Season**: Start and end months



## Data Reference

### Soil Types
- **0**: Sandy
- **1**: Loamy  
- **2**: Clayey
- **3**: Silty
- **4**: Peaty
- **5**: Chalky
- **6**: Saline

### Crop Types
- **0**: Rice
- **1**: Maize
- **2**: Wheat
- **3**: Cotton
- **4**: Barley
- **5**: Sugarcane
- **6**: Other

### Fertilizer Types
- **0**: Diammonium Phosphate
- **1**: Muriate of Potash (MOP)
- **2**: Single Superphosphate (SSP)
- **3**: NPK
- **4**: Zinc Sulphate
- **5**: DAP
- **6**: Urea



### Youtube Link





