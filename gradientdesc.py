import tensorflow as tf
import numpy as np

#tf.set_random_seed(0)

X=[1., 2., 3., 4.]
Y=[1., 3., 5., 7.]

W = tf.Variable([50000.0])

print('step|       cost|       W')

for step in range(1000+1):
  hypothesis = W * X
  cost = tf.reduce_mean(tf.square(hypothesis - Y))

  alpha = 0.01
  gradient = tf.reduce_mean(tf.multiply(tf.multiply(W,X) - Y, X))
  descent = W - tf.multiply(alpha, gradient)
  W.assign(descent)

  if step % 10 == 0:
    print('{:5}|{:10.4f}|{:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))
    print('{}'.format(tf.multiply(W,X)))