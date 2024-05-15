import os
from typing import Tuple
import numpy as np
import pandas as pd
import sys
import time
import configparser

from filter import return_filtered_df

class SlidingWindow:
	def __init__(self, signal: pd.DataFrame, window_size: int, overlap: float) -> None:
		self.signal = signal
		self.window_size = window_size
		self.overlap = overlap

	def create_windows(self) -> np.ndarray:
		signal = self.signal
		window_size = self.window_size
		overlap = self.overlap

		# Calculate the number of windows
		n_windows = int(np.floor((len(signal) - window_size) / (window_size * (1 - overlap))) + 1)

		# Create the windows
		windows = np.zeros((n_windows, window_size, len(signal.axes[1])))
		for i in range(n_windows):
			start = int(i * window_size * (1 - overlap))
			end = start + window_size
			windows[i] = np.array(signal[start:end])

		return windows

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
	windows = sliding_window.create_windows()
	print("Individual time taken by window:", round(time.process_time()-start, 5), "seconds for data size:", len(signal))
	return windows

if __name__ == "__main__":
	index = int(sys.argv[1])
	# signal, _, dump = return_filtered_df(index)
	window_size = int(sys.argv[2])
	overlap = float(sys.argv[3])
	windows = create_sliding_windows(index, window_size, overlap)
	print(windows[0])

