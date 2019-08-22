import tensorflow as tf
""" 
/c/Users/thomasYoung/.virtualenvs/TensorFlow-and-DeepLearning-Tutorial-mzwYOypb/lib/site-packages/tensorflow/python/platform/build_info.py
cuda_version_number = '10.0'
cudnn_version_number = '7'
msvcp_dll_name = 'msvcp140.dll'
nvcuda_dll_name = 'nvcuda.dll'
cudart_dll_name = 'cudart64_100.dll'
cudnn_dll_name = 'cudnn64_7.dll
"""

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())

import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)
  print(x.shape)
  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

shape = (500000, 500000)
# Force execution on CPU
with tf.device("CPU:0"):
  print("On CPU:")
  x = tf.random.uniform(shape)
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform(shape)
    assert x.device.endswith("GPU:0")
    time_matmul(x)