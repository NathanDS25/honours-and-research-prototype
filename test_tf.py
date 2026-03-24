import tensorflow as tf
print("TF version:", tf.__version__)
from transformers import is_tf_available
print("TF available:", is_tf_available())
import transformers
print("Transformers version:", transformers.__version__)
try:
    from transformers.utils import is_keras_available
    print("Keras available:", is_keras_available())
except:
    pass
