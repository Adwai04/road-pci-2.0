import os
import sys
import pandas as pd
import numpy as np
import configparser
import time

from windows import create_sliding_windows
from filter import return_filtered_df

def ten_pt_avg(data, method = 'ISO'):
    sorted_data = np.sort(data.flatten())
    if method == 'ISO':
        tpa_iso = (np.sum(sorted_data[::-1][0:5]) - np.sum(sorted_data[0:5]))/5 
        return tpa_iso
    if method == 'DIN':
        tpa_din = (np.sum(sorted_data[::-1][0:5]) + np.sum(sorted_data[0:5]))/10
        return tpa_din
    else:
        print("wrong method")
        return None 

def peak_to_peak(data):
    sorted_data = np.sort(data.flatten())
    return sorted_data[-1] - sorted_data[0]

def rms(data):
	return np.sqrt(np.mean(data**2))

def get_features(index):
    windows = create_sliding_windows(index)

    start = time.process_time()
    features = [[] for _ in range(len(windows))]
    points = [[] for _ in range(len(windows))]
    for i in range(len(windows)):   
        feature_1 = np.mean(windows[i,:,2])
        feature_2 = np.min(windows[i,:,2])
        feature_3 = np.max(windows[i,:,2])
        feature_4 = peak_to_peak(windows[i,:,2])
        feature_5 = ten_pt_avg(windows[i,:,2])
        feature_6 = rms(windows[i,:,2])
        feature_7 = np.mean(windows[i,:,4])
        lats = windows[i,:,-2]
        longs = windows[i,:,-1]
        coords = list(set(zip(lats, longs)))

        features[i].extend([feature_6, feature_7, feature_2, feature_3, feature_5, feature_4])
        points[i].extend(coords)
    print("Individual time taken by feature:", round(time.process_time()-start, 5), "seconds for data size:", len(windows)*len(windows[0]))
    return features, points


if __name__ == '__main__':
    index = int(sys.argv[1])
    start = time.process_time()
    features, points = get_features(index)
    print("Total time taken:", round(time.process_time()-start, 2), "seconds")
    print(features[0])