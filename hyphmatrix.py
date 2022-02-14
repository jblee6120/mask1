import tensorflow as tf
import numpy as np

data = np.array([
    #X1, X2, X3, y
    [73., 80., 75., 152. ],
    [93., 88., 93., 185. ],
    [89., 91., 90., 180. ],
    [96., 98., 100., 196. ],
    [73., 66., 70., 142. ]
], dtype=np.float32)

learning_rate = 0.000001

X = data[:, :-1]
y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3,1]))
b = tf.Variable(tf.random.normal([1]) * 20)

def predict(X):
    return tf.matmul(X,W) + b

hypho = predict(X)
n_epochs = 30000

for i in range(n_epochs + 1):

    with tf.GradientTape() as tape:
        hypho = predict(X)
        cost = tf.reduce_mean((tf.square(predict(X) - y)))

    W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print('{:5}|{:10.4f}|{}|{}|{}'.format(i,cost.numpy(), W.numpy()[0], W.numpy()[1], W.numpy()[2]))

test = np.array([
    [70., 80., 90.]
], dtype=np.float32)
print(predict(test))