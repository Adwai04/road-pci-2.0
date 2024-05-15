import os
import sys
import pandas as pd
import numpy as np
import configparser
import ast
import pickle
import joblib
import json

from scripts.features import get_features

def read_config():
	config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
	config = configparser.ConfigParser()
	config.read(config_path)
	svm = config.get('Models', 'svm')
	rf = config.get('Models', 'rf')
	#nodes = config.get('GIS', 'nodes')
	return svm, rf

def load_data(index):
	features, points = get_features(index)
	return features, points

def update_values(index):
	features, points = load_data(index)
	velocity = features[:][1]
	svm, rf= read_config()
	svm_path = os.path.join(os.path.dirname(__file__), '..', svm)
	svm_model = joblib.load(svm_path)
	prediction = svm_model.predict(features)

	return velocity, prediction, points

if __name__ == '__main__':
	index = int(sys.argv[1])
	svm, rf = read_config()
	svm_path = os.path.join(os.path.dirname(__file__), '..', svm)
	svm_model = joblib.load(svm_path)
	# with open(rf, 'rb') as f:
	# 	rf_model = pickle.load(f)
	
	features, points = get_features(index)
	velocity = [x[1] for x in features]
	prediction = svm_model.predict(features)
	
	#Save this as a csv with columns: points, velocity, prediction
	
	with open('track_1.csv', 'w') as f:
		f.write('Latitude, Longitude, Velocity, Prediction\n')
		for i in range(len(points)):
			for point in points[i]:
				lat = point[0]
				lng = point[1]
				vel = velocity[i]
				pred = prediction[i]
				f.write(f'{lat}, {lng}, {vel}, {pred}\n')
		f.close()
