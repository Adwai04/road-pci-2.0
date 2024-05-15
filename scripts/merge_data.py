import os
import sys
import configparser
import numpy as np
import pandas as pd

def read_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
    config.read(config_path)
    acc_data = config.get('Paths', 'acc_data_directory')

    return acc_data

def merge():
    acc_path = read_config()
    acc_path = os.path.join(os.path.dirname(__file__), '../', acc_path)
    merged_data = pd.DataFrame()
    for file in os.listdir(acc_path):
        if file.endswith('.csv'):
            acc_data_path = os.path.join(acc_path, file)
            # print(acc_data_path)
            acc_data = pd.read_csv(acc_data_path)
            merged_data = pd.concat([merged_data, acc_data])
        merged_data = merged_data.reset_index(drop=True)

    merged_data.to_csv('merged_data.csv')

merge()

#merged_data.csv is at index 5