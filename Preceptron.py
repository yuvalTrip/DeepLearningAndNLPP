import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data from CSV
data = pd.read_csv('heart-disease-data.csv')

# Data Preprocessing
# Handle missing values
data = data.dropna().reset_index(drop=True) # cleared NaN values and reset id's 

# Feature Engineering
# No specific feature engineering applied in this example

# Split data into features and labels
X = data.drop(columns=['TenYearCHD'])
y = data['TenYearCHD']

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the model with regularization and dropout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val))

# Evaluate model performance on validation data
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)