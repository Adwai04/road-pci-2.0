import os
import sys
import pandas as pd
import configparser
import joblib
import json

from scripts.features import get_features

def read_config(model='rf', data='roughness'):
	config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
	config = configparser.ConfigParser()
	config.read(config_path)
	if model == 'rf':
		model_roughness = config.get('Models', 'rf_roughness')
		model_event = config.get('Models', 'rf_event')
	elif model == 'svm':
		model_roughness = config.get('Models', 'svm_roughness')
		model_event = config.get('Models', 'svm_event')
	#nodes = config.get('GIS', 'nodes')
	if data == 'roughness':
		return model_roughness
	elif data == 'event':
		return model_event


class Predictor:
	def __init__(self, model='rf', data='roughness'):
		self.model_type = model
		self.data_type = data
		self.features = {
			'roughness': ['z_rms', 'Velocity'],
			'event': ['z_rms', 'z_p2p', 'Velocity', 'Prev_Velocity', 'Next_Velocity']
		}
		self.model = None
		self.pred_df = None
		self.final_df = None

	def load_model(self):
		model = read_config(self.model_type, self.data_type)
		model_path = os.path.join(os.path.dirname(__file__), model)
		self.model = joblib.load(model_path)

	def get_pred_df(self, df):
		pred_labels, lats, lngs = [], [], []
		for i in range(len(df)):
			for j in range(len(df['Latitude'][i])):
				pred_labels.append(df['Pred_Label'][i])
				lats.append(df['Latitude'][i][j])
				lngs.append(df['Longitude'][i][j])

		pred_df = pd.DataFrame({
			'Latitude': lats,
			'Longitude': lngs,
			'Pred_Label': pred_labels
		})
		return pred_df

	def predict(self, index):
		self.load_model()
		feature_df = get_features(index, features=self.features[self.data_type])
		test_df = feature_df[self.features[self.data_type]]
		preds = self.model.predict(test_df)
		windowed_pred_df = feature_df[['Latitude', 'Longitude']]
		windowed_pred_df['Pred_Label'] = preds
		self.pred_df = self.get_pred_df(windowed_pred_df)
		return self.pred_df
	
	def get_final_preds(self, select='max'):
		if select == 'max':
			self.final_df = self.pred_df.groupby(
				['Latitude', 'Longitude'], 
				as_index=False
			).agg({'Pred_Label': 'max'})
		else:
			self.final_df = self.pred_df.drop_duplicates(
				subset=['Latitude', 'Longitude'], 
				keep='last',
				ignore_index=True
			)
		return self.final_df


if __name__ == '__main__':
	index = int(sys.argv[1])
	predictor = Predictor(model='rf', data='roughness')
	predictor.predict(index)
	final_df = predictor.get_final_preds()
	print(final_df.head())
	
	with open('track_1.csv', 'w') as f:
		f.write('Latitude, Longitude, Prediction\n')
		for i in range(len(final_df)):
			lat = final_df['Latitude'][i]
			lng = final_df['Longitude'][i]
			pred = final_df['Pred_Label'][i]
			f.write(f'{lat}, {lng}, {pred}\n')
		f.close()
