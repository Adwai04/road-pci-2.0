import os
import sys
import pandas as pd
import configparser

from scripts.predict import return_predictions

if __name__ == '__main__':
	acc_data_path = os.path.join(os.path.dirname(__file__), 'data/acc/29March2024/pickledData/pickled3.pkl')
	filename = 'track_3'
	acc_df = pd.read_pickle(acc_data_path)
	final_df = return_predictions(
		input_df=acc_df, 
		model='rf', 
		pred_type='roughness', \
		remove_duplicate_latlngs=True
	)
	print(final_df.head())
	final_df.to_csv(filename+'.csv')
