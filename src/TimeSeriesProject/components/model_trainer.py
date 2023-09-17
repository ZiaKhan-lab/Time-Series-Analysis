import os
import joblib
import pandas as pd
from pathlib import Path
from TimeSeriesProject import logger
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.TimeSeriesProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_data.drop(['RUL', 'id'], axis=1, inplace=True)
        test_data.drop(['RUL', 'id'], axis=1, inplace=True)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        train_x_lstm = train_x.values.reshape((train_x.shape[0], train_x.shape[1], 1))
        test_x_lstm = test_x.values.reshape((test_x.shape[0], test_x.shape[1], 1))



        self.model.fit(
            train_x_lstm, train_y,
            batch_size=self.config.params_batch_size,
            epochs=self.config.params_epochs,
            verbose=1  
        )


        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
         )

