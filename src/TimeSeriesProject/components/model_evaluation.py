import os
import pandas as pd
import mlflow
import numpy as np
import joblib
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from src.TimeSeriesProject.entity.config_entity import ModelEvaluationConfig
from src.TimeSeriesProject.utils.common import save_json
from tensorflow.keras.models import load_model
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_data.drop(['RUL', 'id'], axis=1, inplace=True)
        model = load_model(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        test_x = test_x.values.reshape((test_x.shape[0], test_x.shape[1], 1))


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            y_pred = (predicted_qualities >= 0.5).astype(int)

            accuracy = accuracy_score(test_y, y_pred)

            (rmse, mae, r2) = self.eval_metrics(test_y, y_pred)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2, "acc": accuracy}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("acc", accuracy)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="CNN")
            else:
                mlflow.sklearn.log_model(model, "model")

    


    
