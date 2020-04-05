import tensorflow as tf 
import tensorflow_hub as hub
from tensorflow.keras import layers

import numpy as np
import os
import time

train_dataset_dir = 'images/train'
val_dataset_dir = 'images/validation'

batch_size = 8
image_res = 224

labels = np.array(sorted(os.walk(train_dataset_dir).next()[1]))


URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL, input_shape=(image_res, image_res,3))


def get_images(dataset_dir):
	#list of image paths and the images' corresponding label ids
	image_names, image_label_ids = [], []

	label_id = 0

	for label in labels:
		label_dir = os.path.join(dataset_dir, label)

		image_paths = os.walk(label_dir).next()


		for image_path in image_paths[2]:
			if image_path.endswith('jpg') or image_path.endswith('jpeg'):
				image_names.append(os.path.join(label_dir, image_path))
				image_label_ids.append(label_id)

		label_id += 1

	return image_names, image_label_ids


def parser(image_name, label_id):
	decoded_image = tf.image.decode_jpeg(tf.io.read_file(image_name), channels=3)
	image = tf.cast(decoded_image, tf.float32)
	image = tf.image.resize(image, (image_res, image_res))/255.0
	return image, label_id


def save_model(model):
	t = time.time()

	export_path_sm = "./{}".format(int(t))
	print(export_path_sm)

	tf.saved_model.save(model, export_path_sm)


training_images, training_labels = get_images(train_dataset_dir)

training_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(training_images), tf.constant(training_labels)))
training_dataset = training_dataset.map(parser)
training_dataset = training_dataset.shuffle(len(training_images))
training_dataset = training_dataset.batch(batch_size)


val_images, val_labels = get_images(val_dataset_dir)

val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(val_images), tf.constant(val_labels)))
val_dataset = val_dataset.map(parser)
val_dataset = val_dataset.shuffle(len(val_images))
val_dataset = val_dataset.batch(batch_size)


feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(
  optimizer='adam', 
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

EPOCHS = 6
history = model.fit(training_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset)

save_model(model)

# image_batch, label_batch = next(iter(training_dataset.take(1)))
# image_batch = image_batch.numpy()
# label_batch = label_batch.numpy()

# predicted_batch = model.predict(image_batch)
# predicted_batch = tf.squeeze(predicted_batch).numpy()
# predicted_ids = np.argmax(predicted_batch, axis=-1)
# predicted_class_names = labels[predicted_ids]
# print(predicted_class_names)

# print("Labels: ", label_batch)
# print("Predicted labels: ", predicted_ids)




#RELOAD
reload_sm_keras = tf.keras.models.load_model(
  '1578826768',
  custom_objects={'KerasLayer': hub.KerasLayer})


# img_name = val_dataset_dir + '/paper/19774.jpg'
# img, label = parser(img_name, 3)

# img = np.expand_dims(img, axis=0)

# predict = reload_sm_keras.predict(img)
# predict =  tf.squeeze(predict).numpy()
# predict_id = np.argmax(predict, axis=-1)
# predicted_class_name = labels[predict_id]
# print(predicted_class_name)

EPOCHS = 3
history = reload_sm_keras.fit(training_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset)

save_model(reload_sm_keras)

# image_batch, label_batch = next(iter(training_dataset.take(1)))
# image_batch = image_batch.numpy()
# label_batch = label_batch.numpy()

# predicted_batch = reload_sm_keras.predict(image_batch)
# predicted_batch = tf.squeeze(predicted_batch).numpy()
# predicted_ids = np.argmax(predicted_batch, axis=-1)
# predicted_class_names = labels[predicted_ids]
# print(predicted_class_names)

# print("Labels: ", label_batch)
# print("Predicted labels: ", predicted_ids)


