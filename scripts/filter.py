import os
from typing import Tuple
import configparser
import time
import numpy as np
import pandas as pd
import sys
from scipy import signal
from scipy.signal import butter, filtfilt

from reorient import return_reoriented_df

class ButterFilter:
	def __init__(self, signal: pd.DataFrame, low_cutoff: float, high_cutoff: float, fs: float, order: int = 5) -> None:
		self.signal = signal
		self.low_cutoff = low_cutoff
		self.high_cutoff = high_cutoff
		self.fs = fs
		self.order = order

	def butter_lowpass(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		signal = self.signal
		
		b, a = butter(self.order, self.low_cutoff / (self.fs / 2), btype='low', analog=False)
		x = filtfilt(b, a, np.array(signal['x_acc']))
		y = filtfilt(b, a, np.array(signal['y_acc']))
		z = filtfilt(b, a, np.array(signal['z_acc']))
		return x, y, z

	def butter_highpass(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		signal = self.signal

		b, a = butter(self.order, self.high_cutoff / (self.fs / 2), btype='high', analog=False)
		x = filtfilt(b, a, np.array(signal['x_acc']))
		y = filtfilt(b, a, np.array(signal['y_acc']))
		z = filtfilt(b, a, np.array(signal['z_acc']))
		return x, y, z

	def butter_bandpass(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		signal = self.signal

		b, a = butter(self.order,[self.low_cutoff/(self.fs / 2) ,self.high_cutoff / (self.fs / 2)], btype='band', analog=False)
		x = filtfilt(b, a, np.array(signal['x_acc']))
		y = filtfilt(b, a, np.array(signal['y_acc']))
		z = filtfilt(b, a, np.array(signal['z_acc']))
		return x, y, z


def read_config() -> Tuple[float, float, float, int]:
	config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
	config = configparser.ConfigParser()
	config.read(config_path)
	low_cutoff = config.get('Parameters', 'LOWCUT')
	high_cutoff = config.get('Parameters', 'HIGHCUT')
	fs = config.get('Parameters', 'SAMPLING_RATE')
	order = config.get('Parameters', 'ORDER')
	return float(low_cutoff), float(high_cutoff), float(fs), int(order)

def butter_filter(signal: pd.DataFrame, low_cutoff: float, high_cutoff: float, fs: float, order: int, filter_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	print(f"Applying {filter_type} filter...")
	butter_filter = ButterFilter(signal, low_cutoff, high_cutoff, fs, order)
	if filter_type == 'lowpass':
		x, y, z = butter_filter.butter_lowpass()
	elif filter_type == 'highpass':
		x, y, z = butter_filter.butter_highpass()
	elif filter_type == 'bandpass':
		x, y, z = butter_filter.butter_bandpass()
	
	return x,y,z



def return_filtered_df(index: int, resample=False) -> pd.DataFrame:
	low_cutoff, high_cutoff, fs, order = read_config()
	signal = return_reoriented_df(index, resample=resample)
	start = time.process_time()
	x_lp, y_lp, z_lp = butter_filter(signal, low_cutoff, high_cutoff, fs, order, filter_type='lowpass')
	x_hp, y_hp, z_hp = butter_filter(signal, low_cutoff, high_cutoff, fs, order, filter_type='highpass')
	x_bp, y_bp, z_bp = butter_filter(signal, low_cutoff, high_cutoff, fs, order, filter_type='bandpass')

	signal_lp = signal.copy()
	signal_hp = signal.copy()
	signal_bp = signal.copy()

	signal_lp['x_acc'] = x_lp
	signal_lp['y_acc'] = y_lp
	signal_lp['z_acc'] = z_lp

	signal_hp['x_acc'] = x_hp
	signal_hp['y_acc'] = y_hp
	signal_hp['z_acc'] = z_hp

	signal_bp['x_acc'] = x_bp
	signal_bp['y_acc'] = y_bp
	signal_bp['z_acc'] = z_bp
	print("Individual time taken by filter:", round(time.process_time()-start, 5), "seconds for data size:", len(signal))
	return signal_lp, signal_hp, signal_bp

if __name__ == '__main__':
	index = int(sys.argv[1])
	signal_lp, signal_hp, signal_bp = return_filtered_df(index)#, 0.5, 20, 50, 5)
	# print(signal_lp['Time'])
	print(signal_lp)
	# print("Highpass filtered signal:", signal_hp)
	# print("Bandpass filtered signal:", signal_bp)
