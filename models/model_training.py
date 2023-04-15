import numpy as np
import time

import PIL.Image as Image
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import datetime

print(tf.__version__)

# Download the dataset
data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = np.array(train_ds.class_names)
print(class_names)

# Rescaling images to [0,1]
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

# Data pre-fetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Feature extractor
feature_extractors = {
  "025": "https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/5",
  "050": "https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/5",
  "100": "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5"
}

depth_multiplier = "050"

feature_extractor_model = feature_extractors[depth_multiplier]
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

# Classification head
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes),
  tf.keras.layers.Softmax()
])

print(model.summary())

# Compile model
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['acc'])

# Create training callbacks for TensorBoard and EarlyStopping
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# Train model for <= 30 epochs
NUM_EPOCHS = 30

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=[tensorboard_callback, early_stopping_callback])

"""## Export and reload your model

Now that you've trained the model, export it as a SavedModel for reusing it later.
"""

print("History: ", history.history)

export_path = "mobilenetv1_{}/{}_val_acc".format(depth_multiplier, round(history.history["val_acc"][-4] * 100, 2))
model.save(export_path)

print("Model saved in:", export_path)

### INT-8 Export
# Reload the model for TF Lite export
reloaded = tf.keras.models.load_model(export_path)

# Representative dataset
def representative_dataset(dataset):

  def _data_gen():
    for img, label in dataset:
      for i in img:
        i = tf.expand_dims(i, 0)
        i = i
        yield [i]

  return _data_gen

converter = tf.lite.TFLiteConverter.from_keras_model(reloaded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset(val_ds)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open(export_path + '/lite/model_uint8.tflite', 'wb') as f:
  f.write(tflite_model)

# get full-precision accuracy
loss, acc = reloaded.evaluate(val_ds)
print(f'FP32 accuracy: {acc * 100:.2f}%')

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

results = []

for data in representative_dataset(val_ds)():
  inp = np.uint8(data[0] * 255)
  interpreter.set_tensor(input_idx, inp)
  interpreter.invoke()
  results.append(np.argmax(interpreter.get_tensor(output_idx).flatten()))

labels = []
for img, label in val_ds:
  for l in label:
    labels.append(tf.get_static_value(l))

acc = np.sum(np.array(results) == np.array(labels)) / len(labels)
print(f'INT8 accuracy: {acc * 100:.2f}%')


### FP-16 Model export
def representative_dataset(dataset):

  def _data_gen():
    for img, label in dataset:
      for i in img:
        i = tf.expand_dims(i, 0)
        yield [i]

  return _data_gen

converter = tf.lite.TFLiteConverter.from_keras_model(reloaded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset(val_ds)
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open(export_path + '/lite/model_fp16.tflite', 'wb') as f:
  f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

results = []

for data in representative_dataset(val_ds)():
  inp = np.float32(data[0])
  interpreter.set_tensor(input_idx, inp)
  interpreter.invoke()
  results.append(np.argmax(interpreter.get_tensor(output_idx).flatten()))

acc = np.sum(np.array(results) == np.array(labels)) / len(labels)
print(f'FP16 accuracy: {acc * 100:.2f}%')