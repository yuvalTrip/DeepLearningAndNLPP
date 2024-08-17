import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the data
data = pd.read_csv("heart-disease-dataV2.csv")

# Handle missing values- Any rows with missing values are removed, and the index is reset to ensure continuity.
data = data.dropna().reset_index(drop=True)

# Preprocessing
X = data.drop(columns=["HeartDisease"])
y = data["HeartDisease"]

# Split data into training and testing sets-80% used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
#The features are standardized to have a mean of 0 and a standard deviation of 1.
#This improves the performance of the neural network by ensuring that all features contribute equally.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for CNN input
#The data is reshaped to add a channel dimension, which is required by the Conv1D layer. 
#The shape changes from (samples, features) to (samples, features, 1).
X_train = np.expand_dims(X_train, axis=2)  # Adding channel dimension
X_test = np.expand_dims(X_test, axis=2)

# Define the CNN model
model = Sequential([
    #A 1D convolutional layer with 64 filters and a kernel size of 3, followed by a ReLU activation function.
    #It processes the input features sequentially.
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Flatten(),#Flattens the output from the convolutional layer to a 1D vector.
    Dense(128, activation='relu'),#A fully connected layer with 128 neurons and a ReLU activation function.
    Dropout(0.5),#A dropout layer with a dropout rate of 0.5 to prevent overfitting.
    Dense(1, activation='sigmoid')# A dense layer with a single neuron and a sigmoid activation function for binary classification.
])

# Compile the model
#other tries - without adam - did not improve, with adam & learning rate - did not improve either.
model.compile(optimizer='adam',# A popular optimizer for training deep learning models.
              loss='binary_crossentropy', #Suitable for binary classification tasks.
              metrics=['accuracy']) #Used to evaluate the model’s performance

# Train the model
#other tries - batch 32, and 64 - did not improve at all
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1) #A validation split of 0.1 means 10% of the training data is used for validation during training to monitor the model's performance.

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
