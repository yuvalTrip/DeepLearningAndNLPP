import numpy as np
import pandas as pd
import tensorflow as tf

def perceptron():
    # Load data from CSV
    data = pd.read_csv('heart-disease-data.csv')
    features = data.iloc[:, :-1]  # Select all columns except the last one as features
    labels = data.iloc[:, -1]     # Select the last column as labels

    # Convert data to numpy arrays
    data_x = features.values
    data_y = labels.values.reshape(-1, 1)  # Reshape labels to match model output shape

    # Define model architecture
    features_count = data_x.shape[1]
    hidden_layer_nodes = 15

    inputs = tf.keras.Input(shape=(features_count,), dtype=tf.float64)
    x = tf.keras.layers.Dense(hidden_layer_nodes, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    # Train the model
    """we started with 10 epoches and get 
    The ultimate epochs number we arrive was 80 with loss of 0.4261, but we saw it converge from epoch 23 to this value.
    
    
    """
    model.fit(data_x, data_y, epochs=10, batch_size=32)

if __name__ == "__main__":
    perceptron()
