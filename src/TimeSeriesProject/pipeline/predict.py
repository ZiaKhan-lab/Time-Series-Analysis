import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","model_trainer", "model.h5"))

        imagename = self.filename
        new_data = pd.read_csv(imagename,sep=" ",header=None)
        new_data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21","sensor22","sensor23"]
        
        new_data_ = new_data.drop(['id',"cycle","op1", "op2", "op3", "sensor1", "sensor5", "sensor6", "sensor10", "sensor16", "sensor9","sensor18", "sensor19", "sensor14", "sensor13", "sensor12", "sensor11",'sensor22', 'sensor23'], axis=1)
                             
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(new_data_)
        scaled_features = pd.DataFrame(scaled_features, columns=new_data_.columns)

        data_test = scaled_features.copy()

        X_test_data_cnn = data_test.values.reshape((data_test.shape[0], data_test.shape[1], 1))

        y_pred_prob = model.predict(X_test_data_cnn)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        label_test_data = y_pred.flatten()
        label = pd.DataFrame(label_test_data)
        data_test['id']=new_data['id']
        data_test['cycle']=new_data['cycle']
        data_test['label']=label
        data_test['label'] = data_test['label'].replace({0: 'Safe', 1: 'Failure'})
        data_test['RUL']=0

        data_test.to_csv(os.path.join("artifacts/model_evaluation/predicted_test_data04.csv"),index = False)

        print(data_test)

