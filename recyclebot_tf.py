import serial, time, os
import numpy as np
from cv2 import *

import tensorflow as tf 
import tensorflow_hub as hub

train_dataset_dir = 'images/train'
val_dataset_dir = 'images/validation'

batch_size = 8
image_res = 224

arduino = serial.Serial('/dev/cu.wchusbserial1420', 115200, timeout = 1)
print("connected")
cam = VideoCapture(1)

labels = np.array(['metal', 'paper', 'plastic', 'trash'])

img_name = 'recyclebot.jpg'


reload_sm_keras = tf.keras.models.load_model(
  '1578826768',
  custom_objects={'KerasLayer': hub.KerasLayer})


def parse(image_name):
	decoded_image = tf.image.decode_jpeg(tf.io.read_file(image_name), channels=3)
	image = tf.cast(decoded_image, tf.float32)
	image = tf.image.resize(image, (image_res, image_res))/255.0
	return image


def get_prediction():

	img = parse(img_name)

	img = np.expand_dims(img, axis=0)

	predict = reload_sm_keras.predict(img)
	predict =  tf.squeeze(predict).numpy()
	predict_id = np.argmax(predict, axis=-1)
	predicted_class_name = labels[predict_id]
	return predicted_class_name


try:
	while True:
		data = arduino.read(arduino.inWaiting())
		print(data)
		if(data == b'1'):
			time.sleep(1)
			s, img = cam.read()
			print("image taken")
			if s:
				if os.path.isfile(img_name):
					os.remove(img_name)
				imwrite("recyclebot.jpg",img)
				prediction = get_prediction()
				arduino.write(prediction.encode('UTF-8'))
			else: print("error")
		time.sleep(1);
except Exception as e:
	print(e)
	print("closed??")
	arduino.close();
	cam.release();
