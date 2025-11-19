import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    df = pd.read_csv("colors.csv")
except FileNotFoundError:
    print("Error: colors.csv not found. Please run generate_dataset.py first.")
    exit()

print("Dataset loaded successfully.")
X = df[['hue', 'saturation', 'value']].values 
y = df['color_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training KNN on {len(X_train)} samples...")

# n_neighbors=5 is usually good, but since we have many points, we can slightly increase it for stability
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "color_model.pkl")
print("\nModel saved to 'color_model.pkl'. Ready for app.py!")