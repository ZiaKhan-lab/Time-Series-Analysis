import os
import tensorflow as tf
from zipfile import ZipFile
import urllib.request as request
from keras.models import Sequential
from keras.layers import LSTM
from src.TimeSeriesProject.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        #self.model = tf.keras.models.Sequential(
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(9,1)))
        # tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(9, 1)),
        # tf.keras.layers.MaxPooling1D(pool_size=2),
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(64, activation='relu')

        self.save_model(path=self.config.base_model_path, model=self.model)


    @staticmethod
    def _prepare_full_model(model, classes):
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="sigmoid"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=1
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
