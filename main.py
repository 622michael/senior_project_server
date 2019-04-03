from flask import Flask
from flask import request
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json

app = Flask(__name__)

def init():
	global model, graph
	model = load_model("./current-model-retrained")
	graph = tf.get_default_graph()

def get_gender_classification(img):
	result = model.predict(img)
	return {
		"Male": float(result[0][0]),
		"Female": float(result[0][1])
	}

@app.route('/', methods=["POST"])
def predict():
	final_shape = (178, 218)
	img = Image.open(request.files["img"]).convert('RGB')
	img = np.array(img)
	img = img[:, :, ::-1].copy() 
	img = cv2.resize(img, final_shape)
	img = np.array([img])
	classification = {}
	with graph.as_default():
		classification["gender"] = get_gender_classification(img)
	return json.dumps(classification)

if __name__ == '__main__':
	init()
	app.run(threaded=True)
