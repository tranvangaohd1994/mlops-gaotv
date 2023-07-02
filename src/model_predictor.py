
import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request, BackgroundTasks
from pandas.util import hash_pandas_object
from pydantic import BaseModel

from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath
import time

rootF = '/home/ubuntu/mlops/mlops-gaotv'
class Data(BaseModel):
    id: str
    rows: list
    columns: list

class ModelPredictor:
    def __init__(self, config_file_path):

        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        self.model = mlflow.sklearn.load_model(model_uri)

    def detect_drift(self, probas) -> int:
        # watch drift between coming requests and training data
        thresh = 0.7
        x_0 = probas[probas[:,0] >=thresh]
        x_1 = probas[probas[:,1] >=thresh]
        lower_samples = len(x_0) + len(x_1)
        if (1.0*lower_samples)/len(probas) < 0.3:
            return 1
        return 0
    def predict(self, data: Data):
        
        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # ModelPredictor.save_request_data(
        #     raw_df, self.prob_config.captured_data_dir, data.id
        # )
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )

        if 'is_drift' in feature_df.columns:
            feature_df = feature_df.drop(columns=['is_drift'])
        if 'batch_id' in feature_df.columns:
            feature_df = feature_df.drop(columns=['batch_id'])
        
        #print('-------------', self.config["prob_id"], data.id)
        if False:
            feature_df_drop = pd.DataFrame(feature_df_drop.to_numpy())
            if self.config["prob_id"] == 'prob-1':
                feature_df_drop[[0, 1, 10, 11, 12]] = feature_df_drop[[0, 1, 10, 11, 12]].astype('int64')
            else:
                feature_df_drop[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]] = feature_df_drop[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]].astype('int64')

        probas = self.model.predict_proba(feature_df)
        prediction = probas.argmax(axis=-1)
        is_drifted = self.detect_drift(probas)

        res = {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }
        return res

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


default_config_path_prob1 = rootF +'/data/model_config/phase-1/prob-1/model-1.yaml'
predictor_prob1 = ModelPredictor(config_file_path=default_config_path_prob1)

default_config_path_prob2 = rootF +'/data/model_config/phase-1/prob-2/model-1.yaml'
predictor_prob2 = ModelPredictor(config_file_path=default_config_path_prob2)


app = FastAPI()

@app.get("/")
def root():
    return {"message": "hello"}

@app.post("/phase-1/prob-1/predict")
async def predict_prob1(data: Data, request: Request):
    response = predictor_prob1.predict(data)
    return response

@app.post("/phase-1/prob-2/predict")
async def predict_prob2(data: Data, request: Request):
    response = predictor_prob2.predict(data)
    return response
