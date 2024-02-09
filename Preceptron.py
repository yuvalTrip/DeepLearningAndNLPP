import numpy as np
import tensorflow as tf

def perceptron():
    features = 15
    hidden_layer_nodes = 10
    x = tf.keras.Input(shape=(features,), dtype=tf.float64)
    y_ = tf.keras.Input(shape=(1,), dtype=tf.float64)

    W1 = tf.Variable(tf.cast(tf.random.truncated_normal([features, hidden_layer_nodes], stddev=0.1), dtype=tf.float64))
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden_layer_nodes], dtype=tf.float64))

    #z1 = tf.add(tf.matmul(x, W1), b1)###

    first=tf.cast(tf.matmul(x, W1), tf.float64)
    z1 = tf.add(first, b1)  # Cast the result of matmul to tf.float64

    a1 = tf.nn.relu(z1)

    W2 = tf.Variable(tf.cast(tf.random.truncated_normal([hidden_layer_nodes, 1], stddev=0.1), dtype=tf.float64))
    b2 = tf.Variable(0.)
    z2 = tf.matmul(a1, W2) + b2
    pred = 1 / (1.0 + tf.exp(-z2))

    loss = tf.reduce_mean(-(y_ * tf.math.log(pred) + (1 - y_) * tf.math.log(1 - pred)))
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = 1 / (1.0 + tf.exp(-tf.matmul(inputs, W1) + b1))
            current_loss = tf.reduce_mean(-(targets * tf.math.log(predictions) + (1 - targets) * tf.math.log(1 - predictions)))
        gradients = tape.gradient(current_loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))

    data_x = np.random.rand(100, features)
    data_y = np.random.randint(2, size=(100, 1))

    for i in range(50000):
        train_step(data_x, data_y)

if __name__ == "__main__":
    perceptron()
