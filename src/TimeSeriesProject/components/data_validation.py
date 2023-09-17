import os
import pandas as pd
from src.TimeSeriesProject import logger
from src.TimeSeriesProject.entity.config_entity import DataValidationConfig

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def RUL_calculator(self,df,df_max_cycles):
        max_cycle = df_max_cycles["cycle"]
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='id', right_index=True)
        result_frame["RUL"] = result_frame["max_cycle"] - result_frame["cycle"]
        result_frame.drop(['max_cycle'], axis=1, inplace=True)
        return result_frame


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            #Traiing Data
            data = pd.read_csv(self.config.unzip_data_dir,sep=" ", header=None)
            data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21","sensor22","sensor23"]
            
            data.drop(['sensor22', 'sensor23'], axis=1, inplace=True)
            data.to_csv('artifacts/data_validation/output_data.csv', index=False)

            jet_id_and_rul = data.groupby(['id'])[["id" ,"cycle"]].max()
            jet_id_and_rul.set_index('id', inplace=True)
            jet_id_and_rul.to_csv('artifacts/data_validation/jet_id_and_rul.csv', index=False)

            #Testing Data
            test_data = pd.read_csv(self.config.unzip_data_dir_2,sep=" ", header=None)
            test_data.columns = ["id","cycle","op1","op2","op3","sensor1","sensor2","sensor3","sensor4","sensor5"
                    ,"sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13"
                    ,"sensor14","sensor15","sensor16","sensor17","sensor18","sensor19"
                    ,"sensor20","sensor21","sensor22","sensor23"]
            
            test_data.drop(['sensor9','sensor22','sensor23'], axis=1, inplace=True)
            test_data.to_csv('artifacts/data_validation/output_test_data.csv', index=False)

            
            jet_id_and_rul_test = test_data.groupby(['id'])[["id" ,"cycle"]].max()
            jet_id_and_rul_test.set_index('id', inplace=True)
            jet_id_and_rul_test.to_csv('artifacts/data_validation/jet_id_and_rul_test.csv', index=False)


            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            data = self.RUL_calculator(data, jet_id_and_rul)
            data.to_csv('artifacts/data_validation/relevent_data.csv', index=False)

            return validation_status
        
        except Exception as e:
            raise e