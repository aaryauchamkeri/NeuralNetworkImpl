import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
Y_train = to_categorical(y_train, 10)
X_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
Y_test = to_categorical(y_test, 10)

W1 = np.random.randn(128, 784) * np.sqrt(2. / 784)
b1 = np.zeros((128, 1))
W2 = np.random.randn(10, 128) * np.sqrt(2. / 128)
b2 = np.zeros((10, 1))

learning_rate = 0.1
epochs = 15


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def cross_entropy(pred, true):
    return -np.sum(true * np.log(pred + 1e-9))


for epoch in range(epochs):
    loss_sum = 0
    correct = 0
    for x, y in zip(X_train, Y_train):
        x = x.reshape(784, 1)
        y = y.reshape(10, 1)

        z1 = np.dot(W1, x) + b1
        a1 = relu(z1)
        z2 = np.dot(W2, a1) + b2
        out = softmax(z2)

        loss_sum += cross_entropy(out, y)

        if np.argmax(out) == np.argmax(y):
            correct += 1

        dz2 = out - y  # 10 x 1
        dW2 = np.dot(dz2, a1.T)  # 10 x 128
        db2 = dz2

        da1 = np.dot(W2.T, dz2)
        dz1 = da1 * relu_deriv(z1)
        dW1 = np.dot(dz1, x.T)
        db1 = dz1

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch+1}: Train Acc = {correct / len(X_train) * 100:.2f}%, Loss = {loss_sum / len(X_train):.4f}")


correct = 0
for x, y in zip(X_test, y_test):
    x = x.reshape(784, 1)
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    out = softmax(z2)
    pred = np.argmax(out)
    if pred == y:
        correct += 1

print(f"\nTest Accuracy: {correct / len(X_test) * 100:.2f}%")
