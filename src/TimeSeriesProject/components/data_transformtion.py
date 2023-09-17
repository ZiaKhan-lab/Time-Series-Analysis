import os
import pandas as pd
from scipy import stats
from TimeSeriesProject import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.TimeSeriesProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        test_data = pd.read_csv(self.config.data_path_2)
        
        jet_relevant_data = data.drop(["cycle", "op1", "op2", "op3", "sensor1", "sensor5", "sensor6","sensor9", "sensor10", "sensor16", "sensor18", "sensor19", "sensor14", "sensor13", "sensor12", "sensor11"], axis=1)
        jet_relevant_test_data = test_data.drop(["cycle", "op1", "op2", "op3", "sensor1", "sensor5", "sensor6","sensor10", "sensor16", "sensor18", "sensor19", "sensor14", "sensor13", "sensor12", "sensor11"], axis=1)
        
        #Transformation Train data
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(jet_relevant_data.drop(['id', 'RUL'], axis=1))
        scaled_features = pd.DataFrame(scaled_features, columns=jet_relevant_data.drop(['id', 'RUL'], axis=1).columns)
        
        scaled_features['id'] = jet_relevant_data['id']
        scaled_features['RUL'] = jet_relevant_data['RUL']

        dt = scaled_features.copy()


        #Transformation test data
        scaled_features_2 = scaler.fit_transform(jet_relevant_test_data.drop(['id'], axis=1))
        scaled_features_2 = pd.DataFrame(scaled_features_2, columns=jet_relevant_test_data.drop(['id'], axis=1).columns)
        scaled_features_2['id'] = jet_relevant_test_data['id']
        test_dt = scaled_features_2.copy()

        
        cycle=30
        dt['label'] = dt['RUL'].apply(lambda x: 1 if x <= cycle else 0)
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(dt)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        test_dt.to_csv(os.path.join(self.config.root_dir, "relevent_test_data.csv"),index = False)
        

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        

