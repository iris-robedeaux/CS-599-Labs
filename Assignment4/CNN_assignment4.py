"""CNN_week9.ipynb

IST597 :- Implementing CNN from scratch
Week 9 Tutorial

Author:- aam35, ir496
"""
import tensorflow as tf
import numpy as np
import time

tf.random.set_seed(1234)
np.random.seed(1234)

#load data
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train[..., None] / 255.0
x_test  = x_test[..., None] / 255.0

y_train = tf.one_hot(y_train, 10)
y_test  = tf.one_hot(y_test, 10)


# weight norm
class WeightNorm(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.initialized = False

    def build(self, input_shape):
        super().build(input_shape)

        # Original kernel created by Keras
        kernel = self.layer.kernel
        k_shape = kernel.shape

        # v parameter
        self.v = self.add_weight(
            name="v",
            shape=k_shape,
            initializer=tf.keras.initializers.RandomNormal(0, 0.05),
            trainable=True,
        )

        # g is scalar per output channel
        self.g = self.add_weight(
            name="g",
            shape=(k_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        self.initialized = True

    def call(self, inputs):
        # Compute W = g * v / ||v||
        v_norm = tf.nn.l2_normalize(
            self.v,
            axis=list(range(len(self.v.shape) - 1))
        )
        W = v_norm * self.g  

        return tf.matmul(inputs, W) + self.layer.bias

    def get_config(self):
        config = super().get_config()
        return config


#model
class CNN(tf.keras.Model):
    def __init__(self, hidden_size=100, output_size=10):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 5, padding="same")
        self.pool = tf.keras.layers.MaxPool2D(2, 2)
        self.flatten = tf.keras.layers.Flatten()

        self.fc1 = tf.keras.layers.Dense(hidden_size)
        self.fc2 = tf.keras.layers.Dense(output_size)

        # Norms
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()

        # WeightNorm layer
        self.wn_fc1 = WeightNorm(
            tf.keras.layers.Dense(hidden_size))

    def call(self, x, norm=None, training=False):
        x = self.conv1(x)
        if norm == "batch":
            x = self.bn1(x, training=training)
        elif norm == "layer":
            x = self.ln1(x)

        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        if norm == "weight":
            x = self.wn_fc1(x, training=training)
        else:
            x = self.fc1(x)
            if norm == "batch":
                x = self.bn2(x, training=training)
            elif norm == "layer":
                x = self.ln2(x)

        x = tf.nn.relu(x)
        return self.fc2(x)


#training utilities
def accuracy_function(logits, labels):
    preds = tf.argmax(logits, axis=1)
    truth = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, truth), tf.float32))

#training step
def train_step(model, x_batch, y_batch, loss_fn, optimizer, norm):
    with tf.GradientTape() as tape:
        logits = model(x_batch, norm=norm, training=True)
        loss = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# training loop
batch_size = 64
hidden_size = 100
learning_rate = 0.001
num_epochs = 4

model = CNN(hidden_size)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Build tf.data dataset
dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(60000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

time_start = time.time()

for epoch in range(num_epochs):
    epoch_losses = []
    for x_batch, y_batch in dataset:
        loss = train_step(model, x_batch, y_batch, loss_fn, optimizer, norm="layer")
        epoch_losses.append(loss.numpy())

    print(f"Epoch {epoch+1}, Loss = {np.mean(epoch_losses):.4f}")

#testing
logits_test = model(x_test, norm="layer", training=False)
accuracy = accuracy_function(logits_test, y_test) * 100
print(f"Test Accuracy = {accuracy.numpy():.2f}%")

time_taken = time.time() - time_start
print(f"\nTotal time: {time_taken:.2f} seconds")
print(f"Per epoch: {time_taken/num_epochs:.2f} seconds")
