import pickle
import numpy as np

import os


script_dir = os.path.dirname(os.path.abspath(__file__))

# importing model with absolute paths
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(script_dir, 'standard_scaler.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(script_dir, 'minmax_scaler.pkl'), 'rb'))

# Test with sample data
test_data = np.array([[40, 50, 50, 40.0, 20, 6.5, 100]]).reshape(1, -1)
scaled = ms.transform(test_data)
final = sc.transform(scaled)
prediction = model.predict(final)

print(f"Test prediction: {prediction[0]}")