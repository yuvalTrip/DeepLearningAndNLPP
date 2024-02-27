import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset into variables X (features) and y (labels)
dataset = pd.read_csv('heart-disease-dataV2.csv')

dataset = dataset.dropna().reset_index(drop=True)

# Check for NaN or infinite values in the dataset
if dataset.isnull().values.any() or not np.isfinite(dataset).all().all():
    print("Dataset contains NaN or infinite values. Please clean the data.")
    exit()

# Split into features (X) and labels (y)
y = dataset['HeartDisease'].values
X = dataset.drop(columns=['HeartDisease']).values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for RNN
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Define RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(64, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Inspect model architecture
print(model.summary())

# Compile the model with gradient clipping, learning rate adjustment, and L2 regularization
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Callback to monitor and get the loss during training
class GradientCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Compute loss manually
        loss_fn = self.model.compiled_loss._losses[0]  # Get the loss function
        with tf.GradientTape() as tape:
            predictions = self.model(X_train_scaled, training=True)
            labels = tf.reshape(y_train, (-1, 1))  # Reshape labels to match predictions shape
            loss = loss_fn(labels, predictions)


# Train the model with gradient monitoring callback
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[GradientCallback()])

