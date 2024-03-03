import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the data
data = pd.read_csv("heart-disease-dataV2.csv")

# Handle missing values
data = data.dropna().reset_index(drop=True)

# Preprocessing
X = data.drop(columns=["HeartDisease"])
y = data["HeartDisease"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for CNN input
X_train = np.expand_dims(X_train, axis=2)  # Adding channel dimension
X_test = np.expand_dims(X_test, axis=2)

# Define the CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
#other tires - without adam - did not improve, with adam & learning rate - did not improve either.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
#other tries - batch 32, and 64 - did not improve at all
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
