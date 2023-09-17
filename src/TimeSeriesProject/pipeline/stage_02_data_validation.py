from TimeSeriesProject.config.configuration import ConfigurationManager
from TimeSeriesProject.components.data_validation import DataValiadtion
from TimeSeriesProject import logger
import pandas as pd


STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()
        df = pd.read_csv('artifacts/data_validation/output_data.csv')
        df_max_cycles = pd.read_csv('artifacts/data_validation/jet_id_and_rul.csv')
        data_validation.RUL_calculator(df,df_max_cycles)
        


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


