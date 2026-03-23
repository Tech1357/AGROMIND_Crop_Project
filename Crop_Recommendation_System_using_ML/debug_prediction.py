import pickle
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
model = pickle.load(open(os.path.join(script_dir, 'model.pkl'), 'rb'))
ms = pickle.load(open(os.path.join(script_dir, 'minmax_scaler.pkl'), 'rb'))
sc = pickle.load(open(os.path.join(script_dir, 'standard_scaler.pkl'), 'rb'))

# Test with exact values from notebook that should give different results
test_cases = [
    [40, 50, 50, 40.0, 20, 6.5, 100],      # Should predict Apple (8)
    [100, 90, 100, 50.0, 90.0, 6.5, 202.0], # Different input
    [10, 10, 10, 15.0, 80.0, 4.5, 10.0]     # Different input
]

print("Testing different scaling approaches:")

for i, test_input in enumerate(test_cases):
    print(f"\nTest case {i+1}: {test_input}")
    features = np.array(test_input).reshape(1, -1)
    
    # Method 1: MinMax then Standard (current app.py)
    try:
        scaled1 = ms.transform(features)
        final1 = sc.transform(scaled1)
        pred1 = model.predict(final1)
        print(f"  MinMax->Standard: {pred1[0]}")
    except Exception as e:
        print(f"  MinMax->Standard: Error - {e}")
    
    # Method 2: Only Standard (notebook approach)
    try:
        final2 = sc.transform(features)
        pred2 = model.predict(final2)
        print(f"  Only Standard: {pred2[0]}")
    except Exception as e:
        print(f"  Only Standard: Error - {e}")
    
    # Method 3: Only MinMax
    try:
        final3 = ms.transform(features)
        pred3 = model.predict(final3)
        print(f"  Only MinMax: {pred3[0]}")
    except Exception as e:
        print(f"  Only MinMax: Error - {e}")