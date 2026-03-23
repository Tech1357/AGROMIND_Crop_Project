from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle
import os
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))

# importing model with absolute paths
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(script_dir, 'standard_scaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(script_dir, 'minmax_scaler.pkl'), 'rb'))

# Crop dictionary - matches the order in the dataset
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

# Input validation ranges
ranges = {
    'Nitrogen': (0, 140, 'kg/ha'),
    'Phosporus': (5, 145, 'kg/ha'),
    'Potassium': (5, 205, 'kg/ha'),
    'Temperature': (8, 43, '°C'),
    'Humidity': (14, 99, '%'),
    'Ph': (3.5, 9.5, ''),
    'Rainfall': (20, 298, 'mm')
}

# creating flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Validate inputs
        inputs = {
            'Nitrogen': N,
            'Phosporus': P,
            'Potassium': K,
            'Temperature': temp,
            'Humidity': humidity,
            'Ph': ph,
            'Rainfall': rainfall
        }
        
        for param, value in inputs.items():
            min_val, max_val, unit = ranges[param]
            if not (min_val <= value <= max_val):
                result = f"Error: {param} must be between {min_val} and {max_val} {unit}. Please enter valid values."
                return render_template('index.html', result=result)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply MinMax then Standard scaling (same as training)
        scaled = ms.transform(single_pred)
        final_features = sc.transform(scaled)
        
        # Get probabilities for top 3 predictions
        probabilities = model.predict_proba(final_features)[0]
        
        # Get top 3 indices (classes are 0-21, but our dict starts at 1)
        top_3_indices = np.argsort(probabilities)[-3:][::-1]  # Sort descending
        
        # Map to crop names (add 1 since classes start from 0 but dict from 1)
        top_3_crops = [crop_dict[idx + 1] for idx in top_3_indices]
        top_3_probs = [probabilities[idx] * 100 for idx in top_3_indices]
        
        result = "Top 3 recommended crops:<br>"
        for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs), 1):
            result += f"{i}. {crop} ({prob:.1f}%)<br>"
            
    except ValueError:
        result = "Error: Please enter valid numeric values for all fields."
    except Exception as e:
        result = f"Error occurred: {str(e)}"
        
    return render_template('index.html', result=result)




# python main
if __name__ == "__main__":
    app.run(debug=True)
