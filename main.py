import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ✅ Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# ✅ Load datasets
training_df = pd.read_csv("D:/FINAL MEDICAL/frontend/training.csv", encoding="ISO-8859-1")
description_df = pd.read_csv("D:/FINAL MEDICAL/frontend/description.csv", encoding="ISO-8859-1")
precaution_df = pd.read_csv("D:/FINAL MEDICAL/frontend/precautions.csv", encoding="ISO-8859-1")
medications_df = pd.read_csv("D:/FINAL MEDICAL/frontend/Medication.csv", encoding="ISO-8859-1")
diet_df = pd.read_csv("D:/FINAL MEDICAL/frontend/Diets.csv", encoding="ISO-8859-1")
workout_df = pd.read_csv("D:/FINAL MEDICAL/frontend/workouts.csv", encoding="ISO-8859-1")

# ✅ Prepare features (X) and target (y)
X = training_df.drop(columns=["Disease"])
y = training_df["Disease"]

# ✅ Encode target labels
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ✅ Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
y_resampled = to_categorical(y_resampled)

# ✅ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ✅ Reshape input for LSTM (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# ✅ Build optimized LSTM Model
model = Sequential([
    Input(shape=(1, X.shape[1])),
    LSTM(128, return_sequences=True, activation="tanh", kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64, activation="tanh", kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu", kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(y_categorical.shape[1], activation="softmax")
])

# ✅ Compile Model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# ✅ Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n🔥 Final Model Accuracy: {accuracy * 100:.2f}% 🔥")

# ✅ Save model, label encoder, and scaler
model.save("lstm_model.keras")
with open("lstm_encoder.pkl", "wb") as encoder_file:
    pickle.dump(y_encoder, encoder_file)
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
print("✅ Model, encoder, and scaler saved successfully!")

# ✅ Function to predict disease and fetch details
def get_predicted_value(symptoms):
    input_data = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1
    input_data = scaler.transform(input_data)
    input_data = np.expand_dims(input_data, axis=1)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    predicted_disease = y_encoder.inverse_transform([predicted_class])[0]
    return predicted_disease

def helper(disease):
    description = description_df.loc[description_df['Disease'] == disease, 'Description'].values
    description = description[0] if len(description) > 0 else "No description available"

    precautions = precaution_df.loc[precaution_df['Disease'] == disease].values[:, 1:].flatten().tolist()
    precautions = precautions if precautions else ["No precautions available"]

    medications = medications_df.loc[medications_df['Disease'] == disease, 'Medications'].tolist()
    medications = medications if medications else ["No medications available"]

    diet = diet_df.loc[diet_df['Disease'] == disease, 'Diet'].tolist()
    diet = diet if diet else ["No diet recommendations available"]

    workout = workout_df.loc[workout_df['Disease'] == disease, 'workout'].tolist()
    workout = workout if workout else ["No workout recommendations available"]

    return description, precautions, medications, diet, workout

# ✅ User Input
symptoms = input("Enter your symptoms (comma-separated): ")
user_symptoms = [s.strip() for s in symptoms.split(',')]

# ✅ Predict disease
predicted_disease = get_predicted_value(user_symptoms)
desc, pre, med, die, wrkout = helper(predicted_disease)

# ✅ Display Results
print("\n================= Predicted Disease ==================")
print(predicted_disease)

print("\n================= Description ==================")
print(desc)

print("\n================= Precautions ==================")
for i, p in enumerate(pre, start=1):
    print(f"{i} : {p}")

print("\n================= Medications ==================")
for i, m in enumerate(med, start=1):
    print(f"{i} : {m}")

print("\n================= Workout ==================")
for i, w in enumerate(wrkout, start=1):
    print(f"{i} : {w}")

print("\n================= Diets ==================")
for i, d in enumerate(die, start=1):
    print(f"{i} : {d}")

