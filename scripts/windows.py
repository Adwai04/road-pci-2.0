import os
from typing import Tuple
import numpy as np
import pandas as pd
import sys
import time
import configparser
from scipy import stats

from filter import return_filtered_df

class SlidingWindow:
	def __init__(self, signal: pd.DataFrame, window_size: int, overlap: float) -> None:
		self.signal = signal
		self.window_size = window_size
		self.overlap = overlap
		self.set_cols = ['z_acc', 'Latitude', 'Longitude', 'Velocity', 'Prev_Velocity', 'Next_Velocity'] #columns to extract from filtered df
		self.window_df = pd.DataFrame()

	def create_windows(self, get_cols=None):
		if get_cols is None:
			get_cols = self.set_cols
		signal_df = self.signal
		window_size = self.window_size
		overlap = self.overlap

		window_df = pd.DataFrame()
		windowed_data = {}
		step_size = int(window_size * (1 - overlap))
		n_windows = (len(signal_df) - window_size) // step_size + 1

		for column in get_cols:
			if column == 'Label':
				print("Add labels using add_label function and mention type")
				continue
			col_vals = signal_df[column].to_numpy()
			window_vals = np.empty((n_windows, window_size))
			for i in range(n_windows):
				start_idx = i*step_size
				window_vals[i] = col_vals[start_idx:start_idx + window_size]
			windowed_data[column] = list(window_vals)
		window_df = pd.DataFrame.from_dict(windowed_data)

		if 'Label' not in self.window_df:
			self.window_df = window_df
		else:
			self.window_df = pd.concat([window_df, self.window_df], axis=1)

	def add_labels(self, labels, type='max'):
		# types: max, mode, avg, min
		assert len(labels) == len(self.signal)
		labels = np.array(labels)
		window_size = self.window_size
		overlap = self.overlap
		step_size = int(window_size * (1 - overlap))
		n_windows = (len(self.signal) - window_size) // step_size + 1

		window_vals = np.empty((n_windows, window_size))
		for i in range(n_windows):
			start_idx = i*step_size
			val = None
			if type == 'max':
				val = labels[start_idx:start_idx + window_size].max()
			elif type == 'mode':
				val = stats.mode(labels[start_idx:start_idx + window_size])
			elif type == 'avg':
				val = labels[start_idx:start_idx + window_size].max()
				val = int(np.ceil(val))
			elif type == 'min':
				val = labels[start_idx:start_idx + window_size].min()
			window_vals[i] = int(val)
		self.window_df['Label'] = list(window_vals)

	def get_windows(self):
		return self.window_df

			

def read_config() -> Tuple[int, float]:
	config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
	config = configparser.ConfigParser()
	config.read(config_path)
	window_size = config.get('Parameters', 'WINDOW_SIZE')
	overlap = config.get('Parameters', 'OVERLAP')
	return int(window_size), float(overlap)

def create_sliding_windows(index: int, window=None, ovlap=None) -> np.ndarray:
	print("Creating sliding windows...")
	window_size, overlap = read_config()
	signal, _, _ = return_filtered_df(index)
	if window is not None:
		window_size = window
	if ovlap is not None:
		overlap = ovlap
	start = time.process_time()
	sliding_window = SlidingWindow(signal, window_size, overlap)
	sliding_window.create_windows()
	window_df = sliding_window.get_windows()
	print("Individual time taken by window:", round(time.process_time()-start, 5), "seconds for data size:", len(signal))
	return window_df

if __name__ == "__main__":
	index = int(sys.argv[1])
	# signal, _, dump = return_filtered_df(index)
	window_size = int(sys.argv[2])
	overlap = float(sys.argv[3])
	window_df = create_sliding_windows(index, window_size, overlap)
	print(window_df.head())

