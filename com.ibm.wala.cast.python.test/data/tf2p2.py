import tensorflow
from tensorflow.python.framework.ops import Tensor

def value_index(a,b):
  return a.value_index + b.value_index

# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/Graph#using_graphs_directly_deprecated
g = tensorflow.Graph()
with g.as_default():
  # Defines operation and tensor in graph
  c = tensorflow.constant(30.0)
  assert c.graph is g

result = value_index(Tensor(g.get_operations()[0], 0, tensorflow.float32), Tensor(g.get_operations()[0], 0, tensorflow.float32))
