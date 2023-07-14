import datetime as dt
import os

import pandas as pd


class ModelManager:
    base_output_path = "src/output"
    tracked_metrics: list = []

    def __init__(self):
        self.training_output_path = f"{self.base_output_path}/training"
        self.inference_output_path = f"{self.base_output_path}/inference"
        # create folder if they dont exist
        if not os.path.exists(self.training_output_path):
            os.makedirs(self.training_output_path)
        if not os.path.exists(self.inference_output_path):
            os.makedirs(self.inference_output_path)

    def create_run_path_training(self, model_name: str):
        time_tag = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        run_path = f"{self.training_output_path}/{model_name}/{time_tag}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        return run_path

    def create_run_path_inference(self, model_name: str):
        time_tag = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        run_path = f"{self.inference_output_path}/{model_name}/{time_tag}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        return run_path

    def _listdir_fullpath(self, d):
        return [os.path.join(d, f) for f in os.listdir(d)]

    def _get_train_runs_model(self, model_name: str):
        """Find the training runs for the selected model or all"""
        print(f"Searching for training runs for {model_name}.")
        train_runs = []
        if os.path.exists(f"src/output/training/{model_name}"):
            search_path = f"src/output/training/{model_name}"
            for train_run in self._listdir_fullpath(search_path):
                train_runs.append((model_name, f"{train_run}"))
        else:
            print(f"No training runs found for model {model_name}.")
        return train_runs

    def _get_train_runs_all(self):
        """Find the training runs for all models"""
        print("Searching for training runs in all models.")
        train_runs = []
        for model_name in os.listdir("src/output/training"):
            model_path = os.path.join("src/output/training", model_name)
            for train_run in self._listdir_fullpath(f"{model_path}"):
                train_runs.append((model_name, f"{train_run}"))
        if not train_runs:
            print("No training runs found at all.")
        return train_runs

    def _load_metrics_to_df(self, path: str):
        try:
            metrics = pd.read_csv(path, index_col=0)
        except:
            raise RuntimeError(f"Unable to load metrics from: {path}.")
        return metrics

    def _get_best_metric_from_df(self, metric: pd.DataFrame, watch_metric: str, greater_is_better: bool):
        if watch_metric not in metric.columns:
            raise RuntimeError(f"Watch metric {watch_metric} not tracked.") # TODO: handle error better
        if greater_is_better:
            best_metric = max(metric[watch_metric])
        else:
            best_metric = min(metric[watch_metric])
        return best_metric

    def _get_best_model_info(self, model_name=None, watch_metric: str = "accuracy", greater_is_better: bool = True):
        best_model_info = {
            "model_name": None,
            "best_metric": None,
            "state_dict_path": None,
            "train_history_path": None,
            "train_args_path": None,
        }
        if model_name is None:
            train_runs = self._get_train_runs_all()
        else:
            train_runs = self._get_train_runs_model(model_name)

        if not train_runs:
            return None
        else:
            for model_name, train_run_path in train_runs:
                train_history_path = f"{train_run_path}/train_history.csv"
                metric = self._load_metrics_to_df(train_history_path)
                # find best metric
                if watch_metric not in metric.columns:
                    raise RuntimeError(f"Watch metric {watch_metric} not tracked, available ({metric.columns}).")

                if greater_is_better:
                    best_metric_run = max(metric[watch_metric])
                    if best_model_info["best_metric"] is None or best_metric_run > best_model_info["best_metric"]:
                        best_model_info["model_name"] = model_name
                        best_model_info["best_metric"] = best_metric_run
                        best_model_info["state_dict_path"] = f"{train_run_path}/best_model.pt"
                        best_model_info["train_history_path"] = f"{train_history_path}"
                        best_model_info["train_args_path"] = f"{train_run_path}/train_args.json"
                else:
                    best_metric_run = min(metric[watch_metric])
                    if best_model_info["best_metric"] is None or best_metric_run < best_model_info["best_metric"]:
                        best_model_info["model_name"] = model_name
                        best_model_info["best_metric"] = best_metric_run
                        best_model_info["state_dict_path"] = f"{train_run_path}/best_model.pt"
                        best_model_info["train_history_path"] = f"{train_history_path}"
                        best_model_info["train_args_path"] = f"{train_run_path}/train_args.json"

            return best_model_info

    def get_all_models(self):
        return os.listdir("src/output/training")


