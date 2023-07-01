import argparse
import logging

import mlflow
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, FeaturesData
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
from sklearn.metrics import accuracy_score


from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig
import yaml


class ModelTrainer:
    EXPERIMENT_NAME = "xgb-1" #"catboost-1"#"xgb-1"

    @staticmethod
    def train_model(prob_config: ProblemConfig, model_params, add_captured_data=False):
        
        run_description = """
            experiment with xgb model , and add data capture labeling with active learning thresh active 0.8
        """
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelTrainer.EXPERIMENT_NAME}"
        )
        #mlflow.note.content = run_description

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        train_x = train_x.to_numpy()
        train_y = train_y.to_numpy()
        logging.info(f"loaded {len(train_x)} samples")
        #add data capture
        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.drop(columns=['is_drift','batch_id'])
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()

            x_active, y_active = ModelTrainer.active_data(captured_x, prob_config)
            logging.info(f'shape of data active: {x_active.shape}, {train_x.shape}')
            
            train_x = np.concatenate((train_x, x_active))
            train_y = np.concatenate((train_y, y_active))
            logging.info(f"added {len(x_active)} captured samples by active")
        
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        test_x = test_x.to_numpy()
        test_y = test_y.to_numpy()

        # create model and train model
        if True: #xgb model
            if len(np.unique(train_y)) == 2:
                objective = "binary:logistic"
            else:
                objective = "multi:softprob"

            model = xgb.XGBClassifier(objective=objective, **model_params)
            model.fit(train_x, train_y)
        
        if False: #catboost
            if prob_config.prob_id == 1: #prob-1
                train_x = pd.DataFrame(train_x)
                train_x[[0, 1, 10, 11, 12]] = train_x[[0, 1, 10, 11, 12]].astype('int64')

                cat_features = [0, 1, 10, 11, 12]
                model = CatBoostClassifier(iterations=200, verbose = False)
                model.fit(train_x, train_y, cat_features)
                
                test_x = pd.DataFrame(test_x)
                test_x[[0, 1, 10, 11, 12]] = test_x[[0, 1, 10, 11, 12]].astype('int64')
            else:
                train_x = pd.DataFrame(train_x)
                train_x[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]] = train_x[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]].astype('int64')

                cat_features = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]
                model = CatBoostClassifier(iterations=200, verbose = False)
                model.fit(train_x, train_y, cat_features)
                
                test_x = pd.DataFrame(test_x)
                test_x[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]] = test_x[[0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]].astype('int64')

        # evaluate
        #test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        predictions = model.predict(test_x)
        auc_score = roc_auc_score(test_y, predictions)
        metrics = {"test_auc": auc_score}
        logging.info(f"metrics: {metrics}")

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        mlflow.end_run()
        logging.info("finish train_model")
    
    @staticmethod
    def active_data(data_capture, prob_config):
        if prob_config.prob_id == 'prob-1':
            config_file_path =  'data/model_config/phase-1/prob-1/model-base.yaml'
        else :
            config_file_path =  'data/model_config/phase-1/prob-2/model-base.yaml'

        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"model-config: {config}")
        model_uri = os.path.join(
            "models:/", config["model_name"], str(config["model_version"])
        )
        base_model = mlflow.sklearn.load_model(model_uri)
        #predict = base_model.predict(data_capture)
        #print(len(predict[predict == 1]), '++++++++++++')
        probas = base_model.predict_proba(data_capture)
        
        thresh = 0.7

        x_active_0 = data_capture[probas[:,0] >=thresh]
        y_active_0 = np.zeros(len(x_active_0))
        x_active_1 = data_capture[probas[:,1] >=thresh]
        y_active_1 = np.ones(len(x_active_1))

        x_active = np.concatenate((x_active_0, x_active_1), axis=0)
        y_active = np.concatenate((y_active_0, y_active_1), axis=0)

        return x_active, y_active


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    ModelTrainer.train_model(
        prob_config, model_config, add_captured_data=args.add_captured_data
    )
