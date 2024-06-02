import os
import sys
import pandas as pd
import configparser
import joblib
import json
import copy

from .features import get_features

def read_config(model='rf', data='roughness'):
	config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
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
	def __init__(self, model='rf', pred_type='roughness'):
		self.model_type = model
		self.pred_type = pred_type
		self.features = {
			'roughness': ['z_rms', 'Velocity'],
			'event': ['z_rms', 'z_p2p', 'Velocity', 'Prev_Velocity', 'Next_Velocity']
		}
		self.model = None
		self.pred_df = None
		self.final_df = None
		print(self.features[self.pred_type])

	def load_model(self):
		model = read_config(self.model_type, self.pred_type)
		model_path = os.path.join(os.path.dirname(__file__), '../', model)
		self.model = joblib.load(model_path)

	def get_pred_df(self, df):
		pred_labels, lats, lngs, vels = [], [], [], []
		for i in range(len(df)):
			for j in range(len(df['Latitude'][i])):
				pred_labels.append(df['Pred_Label'][i])
				vels.append(df['Velocity'][i])
				lats.append(df['Latitude'][i][j])
				lngs.append(df['Longitude'][i][j])

		pred_df = pd.DataFrame({
			'Latitude': lats,
			'Longitude': lngs,
			'Velocity': vels,
			'Pred_Label': pred_labels
		})
		return pred_df

	def predict(self, index, input_df: pd.DataFrame = None):
		self.load_model()
		feature_df = get_features(index=index, input_df=input_df, features=copy.deepcopy(self.features[self.pred_type]))
		test_df = feature_df[self.features[self.pred_type]]
		preds = self.model.predict(test_df)
		windowed_pred_df = feature_df[['Latitude', 'Longitude', 'Velocity']]
		windowed_pred_df['Pred_Label'] = preds
		self.pred_df = self.get_pred_df(windowed_pred_df)
		return self.pred_df
	
	def get_final_preds(self, select='max'): #select for label
		if select == 'max':
			self.final_df = self.pred_df.groupby(
				['Latitude', 'Longitude'], 
				as_index=False
			).agg({'Velocity': 'mean', 'Pred_Label': 'max'})
		else:
			self.final_df = self.pred_df.drop_duplicates(
				subset=['Latitude', 'Longitude'], 
				keep='last',
				ignore_index=True
			)
		return self.final_df
	

def show_data_indices():
	config = configparser.ConfigParser()
	config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
	config.read(config_path)
	acc_path = config.get('Paths', 'acc_data_directory')
	acc_data_path = os.path.join(os.path.dirname(__file__), acc_path)
	print("Acc Data:", acc_data_path)
	for idx, file in enumerate(os.listdir(acc_data_path)):
		print(idx, file)
		
def return_predictions(input_df: pd.DataFrame, index: int = None, model='rf', pred_type='roughness', remove_duplicate_latlngs=True):
	predictor = Predictor(model=model, pred_type=pred_type)
	final_df = predictor.predict(index=None, input_df=input_df)
	if remove_duplicate_latlngs:
		final_df = predictor.get_final_preds()
	final_df.rename(columns = {'Pred_Label': 'Prediction'}, inplace = True)
	return final_df

if __name__ == '__main__':
	index = int(sys.argv[1])
	filename = str(sys.argv[2])
	if index < 0:
		show_data_indices()
		exit()
	if filename[-4:] == '.csv':
		filename = filename[:-4]
	elif filename == '0':
		filename = 'track_1'
	predictor = Predictor(model='rf', data='roughness')
	predictor.predict(index)
	final_df = predictor.get_final_preds()
	final_df.rename(columns = {'Pred_Label': 'Prediction'}, inplace = True)
	print(final_df.head())
	final_df.to_csv(filename+'.csv')
