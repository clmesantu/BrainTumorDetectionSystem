import os
import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Define paths
dataset_path = r'C:\Users\Santhosh S\Downloads\tumor_detection with encryption\dataset'
model_dir = r'C:\Users\Santhosh S\Downloads\tumor_detection with encryption\model'
os.makedirs(model_dir, exist_ok=True)

def load_data(dataset_path):
    images = []
    labels = []
    for label, subdir in enumerate(['negative', 'positive']):
        subdir_path = os.path.join(dataset_path, subdir)
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64)).flatten()
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset using pandas
X, y = load_data(dataset_path)

# Convert to DataFrame for feature columns
X_df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(X.shape[1])])
y_series = pd.Series(y, name='label')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)

# Create a pipeline with a standard scaler and SVM classifier
pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))

# Train the model
pipeline.fit(X_train, y_train)

# Define the model path
MODEL_PATH = os.path.join(model_dir, 'cnn_svm_model.pkl')

# Save the trained model to a file
joblib.dump(pipeline, MODEL_PATH)

print("\nModel trained and saved successfully.")