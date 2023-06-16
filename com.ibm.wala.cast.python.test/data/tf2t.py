import tensorflow as tf

def add(a, b):
  return a + b

nested_row_lengths = [tf.constant([2, 1, 0, 2], tf.int64),tf.constant([2, 0, 3, 1, 1], tf.int64),]
x = tf.keras.layers.Input(shape=[None], dtype=tf.string)
y = tf.RaggedTensor.from_nested_row_lengths(x, nested_row_lengths)
c = add(y,y)
