import tensorflow as tf

def returnedTensor(a, b):
  return a, b


g = tf.Graph()
with g.as_default():
  c = tf.constant(30.0)
  assert c.graph is g
op = g.get_operations()
c, d = returnedTensor(tf.experimental.numpy.ndarray(op[0], 0, tf.float32), tf.experimental.numpy.ndarray(op[0], 0, tf.float32))
