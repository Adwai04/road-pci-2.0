import os
import sys
import pandas as pd
import numpy as np
import configparser
import time
import copy

from .windows import create_sliding_windows

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

def get_features(index: int = None, features=['z_rms', 'Velocity'], training=False, label_df=None, pred_type='roughness', input_df: pd.DataFrame = None):
    columns = copy.deepcopy(features)
    columns.extend(['Latitude', 'Longitude'])
    if training and label_df is None:
        print("No label data provided -> Training mode turned off!!")
        training = False
    window_df = create_sliding_windows(index=index, label_df=label_df, pred_type=pred_type, input_df=input_df)
    feature_df = window_df
    start = time.process_time()

    feature_df['z_rms'] = window_df['z_acc'].apply(lambda x: rms(x))
    feature_df['z_max'] = window_df['z_acc'].apply(lambda x: np.max(x))
    feature_df['z_min'] = window_df['z_acc'].apply(lambda x: np.min(x))
    feature_df['z_mean'] = window_df['z_acc'].apply(lambda x: np.mean(x))
    feature_df['z_variance'] = window_df['z_acc'].apply(lambda x: np.var(x))
    feature_df['z_tpah'] = window_df['z_acc'].apply(lambda x: ten_pt_avg(x))
    feature_df['z_p2p'] = window_df['z_acc'].apply(lambda x: peak_to_peak(x))
    feature_df['Velocity'] = window_df['Velocity'].apply(lambda x: np.mean(x))
    feature_df['Prev_Velocity'] = window_df['Prev_Velocity'].apply(lambda x: np.mean(x))
    feature_df['Next_Velocity'] = window_df['Next_Velocity'].apply(lambda x: np.mean(x))  

    if not training and 'Label' in window_df:
        feature_df = feature_df.drop('Label', axis=1)
    elif training and 'Label' not in window_df:
        print("[ERROR] Labels not added in window_df => training_df with labels failed!!")
    elif training and 'Label' in window_df:
        columns.append('Label')

    try:
        feature_df = feature_df[columns]
    except:
        print(f'[ERROR] {columns} not found')
    # print("Individual time taken by feature:", round(time.process_time()-start, 5), "seconds for data size:", len(window_df)*len(window_df['z_acc'][0]))

    return feature_df


if __name__ == '__main__':
    index = int(sys.argv[1])
    start = time.process_time()
    feature_df = get_features(index)
    print("Total time taken:", round(time.process_time()-start, 2), "seconds")
    print(feature_df.head())