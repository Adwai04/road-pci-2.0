import os
from typing import Tuple
import numpy as np
import pandas as pd
import configparser
from .resample import return_resampled_df
import sys
import time

class EulerRO:
	def __init__(self, acc_data, resampling_freq, tuning_time):
		self.acc_data = acc_data
		self.resampling_freq = resampling_freq
		self.tuning_time = tuning_time

	def calculate_rotation(self):
		ax = self.acc_data['x_acc'][:int(self.resampling_freq*self.tuning_time)].mean()
		ay = self.acc_data['y_acc'][:int(self.resampling_freq*self.tuning_time)].mean()
		az = self.acc_data['z_acc'][:int(self.resampling_freq*self.tuning_time)].mean()

		alpha = np.arctan2(ay, az)
		beta = np.arctan(-ax/np.sqrt(ay**2 + az**2))
		rotation_matrix_alpha = np.array(
			[[1,0,0], 
			[0,np.cos(alpha), -np.sin(alpha)],
			[0, np.sin(alpha), np.cos(alpha)]]
		)
		rotation_matrix_beta = np.array(
			[[np.cos(beta), 0, np.sin(beta)],
			[0,1,0],
			[-np.sin(beta), 0, np.cos(beta)]]
		)
		rotation_matrix_1 = rotation_matrix_beta@rotation_matrix_alpha
		rotated_1 = np.dot(rotation_matrix_1, np.array([ax, ay, az]))
		gamma = np.arctan2(rotated_1[0], rotated_1[1])
		rotation_matrix_gamma = np.array(
			[[np.cos(gamma), -np.sin(gamma), 0], 
			[np.sin(gamma), np.cos(gamma), 0], 
			[0, 0, 1]]
		)
		rotation_matrix = rotation_matrix_gamma@rotation_matrix_1
		data = self.acc_data[['x_acc', 'y_acc', 'z_acc']]
		rotated_data = np.dot(rotation_matrix, np.array(data).T)
		return rotated_data

def read_config():
	config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
	config = configparser.ConfigParser()
	config.read(config_path)
	f_resampling = config.get('Parameters', 'F_RESAMPLING')
	tuning_time = config.get('Parameters', 'TUNING_TIME')
	return float(f_resampling), float(tuning_time)

def merge_df(reoreinted_df, resampled_df):
	resampled_df['x_acc'] = reoreinted_df[0]
	resampled_df['y_acc'] = reoreinted_df[1]
	resampled_df['z_acc'] = reoreinted_df[2]
	return resampled_df

def return_reoriented_df(index=None, re_freq=None, t_time=None, input_df: pd.DataFrame = None):
	f_resampling, tuning_time = read_config()
	if re_freq is not None:
		f_resampling = re_freq
	if t_time is not None:
		tuning_time = t_time
	resampled_df = return_resampled_df(index=index, input_df=input_df)
	start = time.process_time()	
	euler_ro = EulerRO(resampled_df, f_resampling, tuning_time)
	reoreinted_df = euler_ro.calculate_rotation()
	final_df = merge_df(reoreinted_df, resampled_df)
	# print("Individual time taken by resample:", round(time.process_time()-start, 5), "seconds for data size:", len(resampled_df))
	return final_df


if __name__ == '__main__':
	index = int(sys.argv[1])
	resampling_freq = float(sys.argv[2])
	tuning_time = float(sys.argv[3])
	reoreinted_df = return_reoriented_df(index, resampling_freq, tuning_time)
	print(reoreinted_df)