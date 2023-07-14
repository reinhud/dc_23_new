import timm
import torch
from src.utils.model_manager import ModelManager
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmInferer:

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self.model_manager = ModelManager()

    def load_model(
            self,
            watch_metric: str = "accuracy",
            greater_is_better: bool = True
    ):
        """Load best model from training run, default to pretrained with base config."""
        best_model_info = self.model_manager._get_best_model_info(self.model_name, watch_metric, greater_is_better)
        print(f"Loading model: {self.model_name} from state dict: {best_model_info['state_dict_path']}")
        if not best_model_info:
            print("No model found, using pretrained model.")
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=True,
                )
            except:
                raise RuntimeError(f"Unable to build model: {self.model_name} from timm.")
        else:
            try:
                model = timm.create_model(
                    self.model_name,
                    pretrained=False,
                    checkpoint_path=best_model_info["state_dict_path"],
                )
                print(f"Loaded model: {self.model_name} from training run: {best_model_info['model_name']}")
            except:
                raise RuntimeError(f"Unable to build best model: {self.model_name} from train history.")
        return model

    def infer(self, model, image):
        """Infer on a list of pictures."""
        model.eval()    # TODO: should make eval before saving

        # transform pictures
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        # Process PIL image with transforms and add a batch dimension
        x = transform(image).unsqueeze(0)

        # Get the labels from the model config
        labels = model.pretrained_cfg['labels']
        top_k = min(len(labels), 5)

        # infer on pictures
        output = model(x)

        # Apply softmax to get predicted probabilities for each class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        # Grab the values and indices of top 5 predicted classes
        values, indices = torch.topk(probabilities, top_k)

        # Prepare a nice dict of top k predictions
        predictions = [
            {"label": labels[i], "score": v.item()}
            for i, v in zip(indices, values)
        ]

        return predictions