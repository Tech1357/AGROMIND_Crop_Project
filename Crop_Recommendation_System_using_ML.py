# Generated from: Crop_Recommendation_System_using_ML.ipynb
# Converted at: 2026-04-29T14:21:05.769Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np

crop=pd.read_csv("Crop_recommendation.csv")

crop.head()

crop.shape

crop.info()

crop.isnull().sum()

crop.duplicated().sum()

crop.describe()

# Drop the non-numeric 'label' column before calculating correlation
numeric_crop = crop.drop('label', axis=1)

# Calculate and display the correlation matrix
display(numeric_crop.corr())

import seaborn as sns
sns.heatmap(numeric_crop.corr(), annot=True, cbar=True)

crop.label.value_counts()

# Reload the data to get the original DataFrame with the 'label' column
crop_original = pd.read_csv("Crop_recommendation.csv")

# Get the value counts of the 'label' column from the original DataFrame
crop_original['label'].value_counts()

crop_original['label'].unique()

crop_original['label'].unique().size

import matplotlib.pyplot as plt
sns.distplot(crop['N'])
plt.show()

crop_original['label'].unique()

crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
crop_original['crop_num']=  crop_original['label'].map(crop_dict)

crop_original

crop_original.head()

crop_original.crop_num.unique()

crop_original.crop_num.value_counts()

# Load data for training on whole dataset
x = crop_original.drop(['label','crop_num'],axis=1)
y = crop_original['crop_num']

x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train.shape

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
x_scaled = ms.fit_transform(x)

x_train

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_final = sc.fit_transform(x_scaled)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# create instances of all models
models = {
    'Naive Bayes': GaussianNB(var_smoothing=1e-9),
    'Support Vector Machine': SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

model_roc_data = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    ypred = model.predict(x_test)
    
    print(f"{name} with accuracy: {accuracy_score(y_test, ypred):.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, ypred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, 23), yticklabels=range(1, 23))
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Compute ROC data for macro-averaged curve
    y_score = model.predict_proba(x_test)
    y_test_binarized = label_binarize(y_test, classes=range(1, 23))
    
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()
    for i in range(22):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(22)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(22):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 22
    
    model_roc_data[name] = (all_fpr, mean_tpr, auc(all_fpr, mean_tpr))

# Plot macro-averaged ROC curves for all models in one figure
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in model_roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Macro-averaged ROC Curves Comparison for All Models')
plt.legend()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance from Random Forest
importances = model.feature_importances_

# Feature names
feature_names = x.columns

# Create dataframe
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort values
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

feature_importance_df

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)

plt.title("Feature Importance for Crop Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()

# Train RandomForest on whole dataset
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(x_final, y)

# Save the model and scalers
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(ms, open('minmax_scaler.pkl', 'wb'))
pickle.dump(sc, open('standard_scaler.pkl', 'wb'))

print('Model and scalers saved.')

def recommendation(N,P,k,temperature,humidity,ph,rainfal):
    features = np.array([[N,P,k,temperature,humidity,ph,rainfal]])
    # Apply MinMax then Standard scaling (same as training)
    scaled = ms.transform(features)
    transformed_features = sc.transform(scaled)
    # Load current model (RandomForest)
    model = pickle.load(open('model.pkl', 'rb'))
    probabilities = model.predict_proba(transformed_features)[0]
    
    # Crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
        21: "Chickpea", 22: "Coffee"
    }
    
    # Get top 3
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_crops = [crop_dict[idx + 1] for idx in top_3_indices]
    top_3_probs = [probabilities[idx] * 100 for idx in top_3_indices]
    
    print("Top 3 recommended crops:")
    for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs), 1):
        print(f"{i}. {crop} ({prob:.1f}%)")
    
    return top_3_crops[0]  # Return top prediction