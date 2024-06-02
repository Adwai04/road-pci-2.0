import os
import sys
import configparser
import pdb  # Importing pdb for debugging

from datetime import datetime

import pickle
import numpy as np
import pandas as pd
import copy

class StandardPreProcessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
      
    def time_to_seconds(self) -> pd.DataFrame:
        time = self.data['Time']
        time = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f') for x in time]
        time = pd.Series(time)
        time = time.apply(lambda x: x.timestamp())
        self.data['Time'] = time
        return self.data

    def remove_duplicates(self) -> pd.DataFrame:
        self.data = self.data.drop_duplicates(subset=['Time'], keep='first')
        self.data = self.data.reset_index(drop=True)
        return self.data

    def add_cols(self) -> pd.DataFrame:
        self.data['Freq'] = np.concatenate([np.array([0]), 1/np.diff(self.data['Time'])])
        self.data['Time_Diff'] = np.concatenate([np.array([0]), np.round(np.diff(self.data['Time']), 4)])
        return self.data

    def pre_process_data(self) -> pd.DataFrame:
        self.data = self.time_to_seconds()
        self.data = self.remove_duplicates()
        self.data = self.add_cols()
        return self.data

class PreSample:
    def __init__(self, data: pd.DataFrame, index: int, downscale_freq: int, upscale_freq: int) -> None:
        self.data = data
        self.downscale_freq = downscale_freq
        self.upscale_freq = upscale_freq
        self.index = index

    def downscale(self) -> pd.DataFrame:
        i,j = 0, 1
        curr_time_diff = self.data.loc[j, 'Time_Diff']

        while j < len(self.data)-1:
            if curr_time_diff < 1/self.downscale_freq:                
                self.data = self.data.drop(j)
                self.data = self.data.reset_index(drop=True)
                self.data.loc[j, 'Time_Diff'] += curr_time_diff
                curr_time_diff = self.data.loc[j, 'Time_Diff']   
            else:
                i,j = j, j+1
                curr_time_diff = self.data.loc[j, 'Time_Diff']
        self.data['Time_Diff'] = np.concatenate([np.array([0]), np.round(np.diff(self.data['Time']), 4)])
        return self.data

    def upscale(self) -> pd.DataFrame:
        i=1
        while i<len(self.data):
            curr_time_diff = self.data.loc[i, 'Time_Diff']
            if curr_time_diff > 1/self.upscale_freq:
                line = pd.DataFrame({'x_acc': np.nan, 'y_acc': np.nan, 'z_acc': np.nan,'Lat': np.nan, 'Lng': np.nan , 'Time': self.data.loc[i-1, 'Time'] + 1/self.upscale_freq, 'Freq': self.upscale_freq, 'Time_Diff': 1/self.upscale_freq}, index=[i])
                self.data = pd.concat([self.data.iloc[:i], line, self.data.iloc[i:]]).reset_index(drop=True)
                self.data.loc[i+1, 'Time_Diff'] = curr_time_diff - 1/self.upscale_freq
            i += 1

        self.data = self.data.ffill()
        self.data = pd.DataFrame(self.data, columns=['x_acc', 'y_acc', 'z_acc', 'Latitude', 'Longitude', 'Time', 'Freq', 'Time_Diff'])
        return self.data

    # def save_to_pickle(self, index: int) -> None:
    #     pickle_location = read_config()[2]
    #     pickle_location = os.path.join(os.path.dirname(__file__), '../', pickle_location)
    #     with open(pickle_location + str(index) + '.pkl', 'wb') as file:
    #         pickle.dump(self.data, file)


    def pre_process_data(self) -> pd.DataFrame:    
        self.data = self.upscale()
        self.data = self.downscale()
        #self.save_to_pickle(self.index)
        return self.data

def read_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
    config.read(config_path)
    acc_data = config.get('Paths', 'acc_data_directory')
    #label_data = config.get('Paths', 'label_data_directory')
    #pickle_data = config.get('Paths', 'pickle_data_directory')
    downscale_freq = config.get('Parameters', 'DOWNSCALE_FREQ')
    upscale_freq = config.get('Parameters', 'UPSCALE_FREQ')
    pre_process_flag = config.get('Parameters', 'PRE_PROCESS_FLAG')

    if int(pre_process_flag) == 0:
        pre_process_flag = False
    else:
        pre_process_flag = True

    return acc_data, int(downscale_freq), int(upscale_freq), pre_process_flag

def load_data(index=None, load_acc=False):
    acc_path, downscale_freq, upscale_freq, pre_process_flag = read_config()
    if load_acc:
        acc_data_path = os.path.join(os.path.dirname(__file__), '../', acc_path)
    #label_data_path = os.path.join(os.path.dirname(__file__), '../', label_path)
    #pickle_data_path = os.path.join(os.path.dirname(__file__), '../', pickle_path)
        try:
            acc_data = os.listdir(acc_data_path)[index]
            acc_data_path = os.path.join(acc_data_path, acc_data)
            # print(acc_data_path)
            if acc_data_path[-4:] == '.csv':
                acc_data = pd.read_csv(acc_data_path)
            elif acc_data_path[-4:] == '.pkl':
                acc_data = pd.read_pickle(acc_data_path)
            
            # label_data = os.listdir(label_data_path)[index]
            # label_data_path = os.path.join(label_data_path, label_data)
            # label_data = pd.read_csv(label_data_path)

            # acc_data_pickled = os.listdir(acc_data_path)[index]
            # acc_data_pickled_path = os.path.join(acc_data_path, acc_data_pickled)
            # acc_data_pickled = pd.read_pickle(acc_data_pickled_path)

            return acc_data, downscale_freq, upscale_freq, pre_process_flag

        except Exception as e:
            print(e)
            return None, None, None, None
    else:
        return None, downscale_freq, upscale_freq, pre_process_flag

def pre_process(index=None, input_df: pd.DataFrame = None):
    if input_df is None:
        acc_data, downscale_freq, upscale_freq, pre_process_flag = load_data(index=index, load_acc=True)
    else:
        acc_data = copy.deepcopy(input_df)
        _, downscale_freq, upscale_freq, pre_process_flag = load_data(load_acc=False)
    if 'accTime' in acc_data and 'Time' not in acc_data:
        acc_data.rename(columns = {'accTime': 'Time'}, inplace = True) 
    if 'Lng' in acc_data and 'Longitude' not in acc_data:
        acc_data.rename(columns = {'Lng': 'Longitude'}, inplace = True) 

    if not pre_process_flag:
        return acc_data

    acc_data = StandardPreProcessor(acc_data).pre_process_data()
    acc_data = PreSample(acc_data, index, downscale_freq, upscale_freq).pre_process_data()
    acc_data['Time'] = acc_data['Time'].apply(lambda x: x - acc_data['Time'].iloc[0])

    return acc_data


if __name__ == '__main__':
    file_type = sys.argv[1]
    index = int(sys.argv[2])
    
    if file_type:
        # acc_data, label_data = pre_process(index)
        acc_data = pre_process(index)
        # print(acc_data.head())
        # print(label_data)
            
        
