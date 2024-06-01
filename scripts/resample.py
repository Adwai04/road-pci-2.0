import os
import sys
from typing import Tuple
import configparser
from scipy.interpolate import BPoly
import pandas as pd
import numpy as np
from scipy.linalg import eigh
from geopy import distance
import time
from datetime import datetime

from .pre_process import pre_process

class BernsteinResample:
    def __init__(self, signal: pd.DataFrame, f_resampling: float) -> None:

        self.signal = signal
        self.f_resampling = f_resampling
    
    def convert_signal(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.signal['x_acc']).reshape(-1, 1),
            np.array(self.signal['y_acc']).reshape(-1, 1),
            np.array(self.signal['z_acc']).reshape(-1, 1),
            np.array(self.signal['Time'])
        )
    
    def poly_fit(self) -> Tuple[BPoly, BPoly, BPoly]:
        x_vals, y_vals, z_vals, time = self.convert_signal()
        return (
            BPoly.from_derivatives(time, x_vals),
            BPoly.from_derivatives(time, y_vals),
            BPoly.from_derivatives(time, z_vals)
        )

    def resample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_poly, y_poly, z_poly = self.poly_fit()
        
        t_np = np.array(self.signal['Time'])
        t_new = np.linspace(0, t_np[-1], int(self.f_resampling*t_np[-1]))

        return (
            x_poly(t_new),
            y_poly(t_new),
            z_poly(t_new),
            t_new
        )

def read_config():
    config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    f_resampling = config.get('Parameters', 'F_RESAMPLING')
    return float(f_resampling)

def resampled_dataframe(signal: pd.DataFrame, f_resampling: float) -> pd.DataFrame:
    bernstein_resample = BernsteinResample(signal, f_resampling)
    x_resampled, y_resampled, z_resampled, t_resampled = bernstein_resample.resample()
    resampled_signal = pd.DataFrame({
        'x_acc': x_resampled,
        'y_acc': y_resampled,
        'z_acc': z_resampled,
        'Time': t_resampled
    })
    return resampled_signal


def get_distance(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.array([float(distance.geodesic((df['Latitude'].iloc[i-1], df['Longitude'].iloc[i-1]),
                      (df['Latitude'].iloc[i], df['Longitude'].iloc[i])).meters)
             for i in range(1, len(df))])
    distances = np.insert(distances, 0,0)
    return distances

def get_velocity (df: pd.DataFrame) -> np.ndarray:
    p_time, p_vel, prev, prev_prev = 0, 0, 0, 0
    velocity = np.zeros(len(df))
    prev_velocity = np.zeros(len(df))
    next_velocity = np.zeros(len(df))
    for i in range (0, len(df)):
        if df['Distance'][i] != 0:
            prev_velocity[prev:i] = p_vel
            vel = df['Distance'][i]/(df['Time'][i]- p_time)
            velocity[prev:i] = vel
            next_velocity[prev_prev: prev] = vel
            prev_prev = prev
            p_time, p_vel, prev = df['Time'][i], vel, i

    velocity[prev: len(df)] = p_vel
    return prev_velocity, velocity, next_velocity
    
def merge_df(signal: pd.DataFrame, resampled_dataframe: pd.DataFrame) -> pd.DataFrame:
    signal['Distance'] = get_distance(signal)
    signal['Prev_Velocity'], signal['Velocity'], signal['Next_Velocity'] = get_velocity(signal)
    merged_df = pd.merge_asof(resampled_dataframe, signal[['Time', 'Velocity', 'Prev_Velocity', 'Next_Velocity', 'Latitude', 'Longitude']], on='Time', direction='nearest')
    return merged_df

def time_to_seconds(df) -> pd.DataFrame:
    time = df['Time']
    time = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f') for x in time]
    time = pd.Series(time)
    time = time.apply(lambda x: x.timestamp())
    df['Time'] = time
    return df

def return_resampled_df(index: int, f=None) -> pd.DataFrame:
    start = time.process_time()
    acc_data = pre_process(index)
    # print("Individual time taken by pre-process:", round(time.process_time()-start, 2), "seconds for data size:", len(acc_data))
    start = time.process_time()
    f_resampling = read_config()
    if f is not None:
        f_resampling = f
    resampled_acc_data = resampled_dataframe(acc_data, f_resampling)
    acc_data_final = merge_df(acc_data, resampled_acc_data)
    # print("Individual time taken by resample:", round(time.process_time()-start, 2), "seconds for data size:", len(acc_data))
    return acc_data_final

if __name__ == '__main__':
    index = int(sys.argv[1])
    f_resampling = int(sys.argv[2])
    resampled_df = return_resampled_df(index, f_resampling)
    print(resampled_df.head())
   