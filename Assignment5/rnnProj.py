#author: ir496
import numpy as np
import matplotlib.pyplot as plt
import random
import deeplake
import tensorflow as tf
from tensorflow.keras import layers

#seeds
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

#data
ds_train = deeplake.load('hub://activeloop/not-mnist-large')
ds_test  = deeplake.load('hub://activeloop/not-mnist-small')

X_train = ds_train.images.numpy().astype(np.float32) / 255.0
X_test  = ds_test.images.numpy().astype(np.float32) / 255.0
y_train = ds_train.labels.numpy().astype(np.int32)
y_test  = ds_test.labels.numpy().astype(np.int32)

X_train = X_train.reshape((-1, 28, 28))
X_test  = X_test.reshape((-1, 28, 28))
num_classes = len(np.unique(y_train))

#gru_cell
class BasicGRUCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        inp = input_shape[-1]
        u = self.units
        self.W = self.add_weight(shape=(inp, 3*u), name="W")
        self.U = self.add_weight(shape=(u, 3*u), name="U")
        self.b = self.add_weight(shape=(3*u,), name="b")
        super().build(input_shape)

    def call(self, x, states):
        h = states[0]
        u = self.units

        Wx = x @ self.W
        Uh = h @ self.U
        Wx_z, Wx_r, Wx_h = tf.split(Wx, 3, axis=1)
        Uh_z, Uh_r, Uh_h = tf.split(Uh, 3, axis=1)
        b_z, b_r, b_h = self.b[:u], self.b[u:2*u], self.b[2*u:]

        z_gate = tf.sigmoid(Wx_z + Uh_z + b_z)
        r_gate = tf.sigmoid(Wx_r + Uh_r + b_r)
        h_tilde = tf.tanh(Wx_h + r_gate * Uh_h + b_h)

        h_new = (1 - z_gate) * h + z_gate * h_tilde
        return h_new, [h_new]

#mgu_cell
class BasicMGUCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        inp = input_shape[-1]
        u = self.units
        self.W = self.add_weight(shape=(inp, 2*u), name="W")
        self.U = self.add_weight(shape=(u, 2*u), name="U")
        self.b = self.add_weight(shape=(2*u,), name="b")
        super().build(input_shape)

    def call(self, x, states):
        h = states[0]
        u = self.units

        Wx = x @ self.W
        Uh = h @ self.U
        Wx_f, Wx_h = tf.split(Wx, 2, axis=1)
        Uh_f, Uh_h = tf.split(Uh, 2, axis=1)
        b_f, b_h = self.b[:u], self.b[u:]

        f_gate = tf.sigmoid(Wx_f + Uh_f + b_f)
        h_tilde = tf.tanh(Wx_h + f_gate * Uh_h + b_h)

        h_new = (1 - f_gate) * h + f_gate * h_tilde
        return h_new, [h_new]

#build_model
def build_rnn(cell_type, units, layers_num, num_classes):
    cell_cls = BasicGRUCell if cell_type == "GRU" else BasicMGUCell
    cells = [cell_cls(units) for _ in range(layers_num)]
    rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(cells) if layers_num > 1 else cells[0])
    inp = tf.keras.Input(shape=(28, 28))
    h = rnn(inp)
    out = tf.keras.layers.Dense(num_classes)(h)
    return tf.keras.Model(inp, out)

#make_ds
def make_ds(X, y, batch=256, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

#train_model
def train_model(model, train_ds, test_ds, epochs):
    opt = tf.keras.optimizers.Adam(1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_hist, test_hist = [], []

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_acc(xb, yb):
        logits = model(xb, training=False)
        pred = tf.argmax(logits, axis=1, output_type=yb.dtype)
        return tf.reduce_sum(tf.cast(pred == yb, tf.int32)), tf.size(yb)

    for epoch in range(epochs):
        for xb, yb in train_ds:
            train_step(xb, yb)

        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0

        for xb, yb in train_ds:
            c, t = eval_acc(xb, yb)
            correct_train += c
            total_train += t

        for xb, yb in test_ds:
            c, t = eval_acc(xb, yb)
            correct_test += c
            total_test += t

        train_acc = correct_train / total_train
        test_acc = correct_test / total_test
        train_hist.append(train_acc.numpy())
        test_hist.append(test_acc.numpy())

        print(f"Epoch {epoch+1}/{epochs}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return train_hist, test_hist

#run
def run(cell_type, trials=3, units_list=None, layer_list=None, epochs=4):
    if units_list is None: units_list = [64, 128, 256]
    if layer_list is None: layer_list = [1, 2, 3]

    train_ds = make_ds(X_train, y_train)
    test_ds = make_ds(X_test, y_test, shuffle=False)
    results = {}

    for t in range(trials):
        units = units_list[t % len(units_list)]
        layers_num = layer_list[t % len(layer_list)]
        name = f"{cell_type}_trial_{t+1}"
        print(f"\n{name}  units={units}  layers={layers_num}")

        model = build_rnn(cell_type, units, layers_num, num_classes)
        train_hist, test_hist = train_model(model, train_ds, test_ds, epochs)
        err = 1 - test_hist[-1]

        results[name] = {
            "units": units,
            "layers": layers_num,
            "train": train_hist,
            "test": test_hist,
            "error": err
        }
        print(f"final test accuracy={test_hist[-1]:.4f}  error={err:.4f}")

    return results

#experiments
gru_results = run("GRU", trials=3, epochs=10)
mgu_results = run("MGU", trials=3, epochs=10)
all_results = {**gru_results, **mgu_results}
