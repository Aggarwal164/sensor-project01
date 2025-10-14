import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustumExceptions
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    artifact_folder : str = os.path.join(artifact_folder)

class DataIngestionClass():
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig
        self.utils = MainUtils

    def export_collection_and_data_frame(self,collection_name, db_name):

        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = MongoClient[db_name][collection_name]

            df = pd.DataFrame(list(collection.find))

            if 'id' in df.columns.to_list():
                df = df.drop(columns=['id'], axis=1)

            df.replace({'na': np.nan}, inplace=True)

            return df
        except Exception as e:
            raise CustumExceptions(e, sys)
        
    def export_data_into_features_store_file_path(self)-> pd.DataFrame:

        try:
            logging.info(f'exporting data from mongodb')
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path, exist_ok=True)

            sensor_data = self.export_collection_and_data_frame(
                collection_name=MONGO_DATABASE_NAME,
                db_name=MONGO_DATABASE_NAME
            )

            logging.info(f'saving exported data into feature store file path:{raw_file_path}')

            feature_store_file_path = os.path.join(raw_file_path,'wafer_fault.csv')

            sensor_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path

        except Exception as e:
            raise CustumExceptions(e, sys)

    def initiate_data_ingestion(self)-> Path:
        logging.info('entered initiated_data_ingestion method of data_ingestion_class')

        try:
            feature_store_file_path = self.export_data_into_features_store_file_path()

            logging.info('got the data from mongodb')

            logging.info('exited initiate_data_ingestion method of data ingestion class')

            return feature_store_file_path
        
        except Exception as e:
            raise CustumExceptions(e, sys)
            

                                             
                                              