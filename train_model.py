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
print(f"Dataset contains {len(df)} samples.")
print("\nFirst 5 rows of the dataset:")
print(df.head())


# X contains the features (the input for the model)
# .values converts the DataFrame to a simple NumPy array, removing the feature names
X = df[['hue', 'saturation', 'value']].values 
# y contains the target label (what the model should predict)
y = df['color_name']

# Split the data into a training set and a testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

# Choose and Train the Model 
print("\nTraining the K-Nearest Neighbors (KNN) model...")

# KNN classifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the model using our training data
model.fit(X_train, y_train)

print("Model training complete.")

# Evaluate the Model's Performance 
print("\nEvaluating model performance on the test data...")

# Use the trained model to make predictions on the test set (data it has never seen)
y_pred = model.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# showing precision, recall, and f1-score for each color
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the Trained Model
model_filename = "color_model.pkl"
print(f"\nSaving the trained model to '{model_filename}'...")

joblib.dump(model, model_filename)

print("Model saved successfully. You are now ready for the final step!")