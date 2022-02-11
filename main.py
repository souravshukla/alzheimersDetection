from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import numpy as np

img_size = 128
def process_images(image_path, img_size = img_size):

  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels = 3)
  img = tf.image.convert_image_dtype(img,tf.float32)
  img = tf.image.resize(img, size = [img_size,img_size])

  return img

batch_size = 32

def create_data_batches(X,y=None,batch_size=batch_size,valid_data=False,test_data=False):
  if test_data:
    print('Creating test data batches....')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_images).batch(batch_size)
    return data_batch

  elif valid_data:
    print('Creating valiation data batches....')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data_batch = data.map(get_image_label).batch(batch_size)
    return data_batch

  else:
    print("Creating training data batches....")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data = data.shuffle(buffer_size = len(X))
    data = data.map(get_image_label)
    data_batch = data.batch(batch_size)
  return data_batch

test_path = 'dataset\t'
test_filenames = [test_path + fname for fname in os.listdir(test_path)]

test_data = create_data_batches(test_filenames[0], test_data=True)
print(test_filenames[0])
model = load_model("alzheinet.h5")

prediction = model.predict(test_data)

print(np.max[prediction])