import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import configparser

from features import get_features


def get_test_path_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), '../', 'config.ini')
    config.read(config_path)
    test_data = config.get('Paths', 'test_data_directory')
    test_data_path = os.path.join(os.path.dirname(__file__), '../', test_data)
    return test_data_path


class TestModel:
    def __init__(self, model_path, features=['z_acc', 'velocity']):
        self.model = None
        self.model_path = model_path
        self.feature_list = features
        self.test_data = None

    def load_model(self):
        self.model = joblib.load(self.model_path)
        print("Model loaded")
        return
    
    def get_data_index(self, data_name):
        filename = os.path.basename(data_name)
        directory = get_test_path_config()
        files = os.listdir(directory)
        try:
            index = files.index(filename)
            return index
        except ValueError:
            print("No such file in test data directory with filename:", filename)
            return -1
        
    def load_test_data(self, index):
        test_directory = get_test_path_config()
        try:
            test_data = os.listdir(test_directory)[index]
            test_data_path = os.path.join(test_directory, test_data)
            if test_data_path[-4:] == '.csv':
                self.test_data = pd.read_csv(test_data_path)
            elif test_data_path[-4:] == '.pkl':
                self.test_data = pd.read_pickle(test_data_path)

            return self.test_data
        except Exception as e:
            print(e)
            return None
        
    def get_accuracy(self, pred_label):
        true_label = self.test_data['true_label'].to_numpy()
        assert len(pred_label) == len(true_label)
        diff = np.array(pred_label)-np.array(true_label)
        num_diff = np.where(diff!=0)[0]
        accuracy = 1-(num_diff/len(true_label))
        accuracy = round(accuracy*100, 2)
        return accuracy

    def get_predictions(self, data_index, get_true_label=True):
        self.load_model()
        test_df = get_features(data_index)
        pred_df = None #pd.DataFrame()
        
        #pred_df = predict(test_df)

        accuracy = self.get_accuracy(pred_df['pred_label'].to_numpy())

        cols = ['Time', 'z_acc', 'velocity', 'pred_label', 'true_label']
        if get_true_label:
            pred_df = pred_df[cols[:-1]]
        else:
            pred_df = pred_df[cols]

        return pred_df, accuracy
    

class Performance:
    def __init__(self, test_bench, models):
        self.test_bench = test_bench
        self.models = models
        self.isUniqueModel = False
        if len(self.models) == 1:
            self.isUniqueModel = True
        self.pred_data = [[pd.DataFrame() for _ in range(len(test_bench))] for _ in range(len(models))]
        self.accuracy = np.zeros((len(models), len(test_bench)))

    def pre_report(self):
        for i, model in enumerate(self.models):
            for j, test_data_idx in enumerate(self.test_bench):
                test_obj = TestModel(model_path=model)
                self.pred_data[i][j], self.accuracy[i, j] = test_obj.get_predictions(test_data_idx, get_true_label=True)

    def gen_report(self, filename='report.txt', compare=True, plot=False):
        #can use comet_ml
        if not compare:
            models = self.models[0] #only the 1st model

        self.pre_report()

        with open(filename, "w") as f:
            for i in range(len(self.test_bench)):
                for j in range(len(models)):
                    text = 'Accuracy of Model: {j} on Test Data: {j} is = {self.accuracy[i, j]}%'
                    f.write(text+'\n')
            f.close()

        if plot:
            fig, ax = plt.subplots(len(self.test_bench), len(self.models))
            for i in range(len(self.test_bench)):
                for j in range(len(models)):
                    data = self.pred_data[i][j]
                    time = data['Time'].to_numpy()
                    z_acc = data['z_acc'].to_numpy()
                    vel = data['Velocity'].to_numpy()
                    pred_label = data['pred_label'].to_numpy()
                    true_lable = data['true_label'].to_numpy()
                    ax[i, j].plot(time, z_acc, label='z_acc')
                    ax[i, j].plot(time, vel, label='velocity')
                    ax[i, j].plot(time, true_lable, label='true_label')
                    ax[i, j].plot(time, pred_label, label='pred_label')
                    ax[i, j].set(xlabel='Time')
                    ax[i, j].set_title('Model: ', j, '-> Test: ', i, ' : Accuracy=', str(self.accuracy[i, j]), '%')
            fig.suptitle('Performance Report Plots')
        return


if __name__ == '__main__':
    test_bench = None
    models = None
    reportObj = Performance(test_bench, models)
    reportObj.gen_report(compare=True, plot=True)



