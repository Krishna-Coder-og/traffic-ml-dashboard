import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("45_traffic_accidents.csv")
X = df.drop(["severity_1to3", "id"], axis=1)
y = df["severity_1to3"]
X = pd.get_dummies(X, drop_first=True)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y) # Training on all data for deployment the model

joblib.dump(rf, 'model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
print("Model created.")
