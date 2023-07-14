import os

import numpy as np
import timm
import torch
from PIL import Image
from src.data.datasets.coin_data import CoinData
from src.utils.model_manager import ModelManager
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

coin_data = CoinData()

class TimmInferer:

    def __init__(
        self,
    ) -> None:
        self.model_manager = ModelManager()

    def load_model(
            self,
            model_name=None,
            watch_metric: str = "accuracy",
            greater_is_better: bool = True
    ):
        """Load best model from training run, default to pretrained with base config."""
        best_model_info = self.model_manager._get_best_model_info(model_name, watch_metric, greater_is_better)
        if not best_model_info:
            raise RuntimeError(f"Unable to build model: {model_name}, no training runs detected. Available model: {self.model_manager.get_all_models()}")
        else:
            try:
                checkpoint = torch.load(best_model_info['state_dict_path'])
                # use number of classes that were used in the transfer learning
                num_classes = checkpoint["model_state_dict"]["fc.weight"].shape[0]
                model = timm.create_model(
                    best_model_info["model_name"],
                    pretrained=False,
                    num_classes=num_classes,
                )

                # load state dict
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded model: {best_model_info['model_name']} from training run: {best_model_info['state_dict_path']}")
            except:
                raise RuntimeError(f"Unable to build best model: {model_name} from train history.")
        return model

    def infer(self, model, images, topk: False):
        """Infer on a list of pictures."""
        model.eval()    # TODO: should make eval before saving

        # transform pictures
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        # Process PIL image with transforms and add a batch dimension
        x = torch.stack([transform(image) for image in images], dim=0)

        # Get the labels from the model config
        labels = list(coin_data.class_to_idx.keys())

        # infer on pictures
        output = model(x)

        # Apply softmax to get predicted probabilities for each class
        probabilities = torch.nn.functional.softmax(output, dim=1)

        predictions = []
        if topk:
            # Grab the values and indices of top k predicted classes
            for proba in probabilities:
                v, i = torch.topk(proba, topk, dim=0)
                pred = [{"label": labels[i.item()], "scores": round(v.item(), 4)} for i, v in zip(i, v)]
                predictions.append(pred)
        else:
            for proba in probabilities:
                v, i = torch.max(proba, dim=0)
                pred = [{"label": labels[i.item()], "scores": round(v.item(), 4)}]
                predictions.append(pred)

        return predictions

    def save_predictions(self, predictions):
        """Save predictions to json."""



    def infer_from_file(self, file_path, model, images, topk: False):
        """Infer on a single picture."""
        if not os.exists(file_path):
            raise RuntimeError(f"File: {file_path} does not exist.")
        else:
            images = [Image.open(f"{file_path}/{img_path}") for img_path in os.listdir(file_path)]

        predictions = self.infer(model, images, topk)

        return predictions