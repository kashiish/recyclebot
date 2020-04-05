import serial, time, os
from cv2 import *
from clarifai.rest import ClarifaiApp, FeedbackInfo, Image as ClImage

app = ClarifaiApp(api_key='cbe43a6a33714963844812816ef1ad14')
model = app.models.get('recyclables')

arduino = serial.Serial('/dev/cu.wchusbserial1420', 115200, timeout = 1)
print("connected")
cam = VideoCapture(1)


try:
	while True:
		data = arduino.read(arduino.inWaiting())
		print(data)
		if(data == b'1'):
			time.sleep(1)
			s, img = cam.read()
			print("image taken")
			if s:
				if os.path.isfile("recyclebot.jpg"):
					os.remove("recyclebot.jpg")
				imwrite("recyclebot.jpg",img)
				response = model.predict_by_filename(os.path.abspath("recyclebot.jpg"))
				concept = response["outputs"][0]["data"]["concepts"][0]
				print(concept["value"])
				print(concept["name"])
				arduino.write(concept["name"].encode('UTF-8'))
			else: print("error")
		time.sleep(1);
except Exception as e:
	print(e)
	print("closed??")
	arduino.close();
	cam.release();
