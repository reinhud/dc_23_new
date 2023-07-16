from fastapi import APIRouter, HTTPException, status


from pytorch_service.src.data.datasets.coin_data import CoinData
from pytorch_service.src.training.config.train_config import TrainConfig
from pytorch_service.src.training.training import Training
from pytorch_service.src.utils.model_manager import ModelManager

import api

router = APIRouter()

model_manager = ModelManager()

@router.get("/models", name="Get all trained models")
async def get_all_models():
    return {
        "All models trained on coin data": model_manager.get_all_models()
    }


# get best model (model_name, metric, k)
# such erst nach modelname, bei default such nach allen
# suche nach metric 
@router.get("/best_model", name="Get best model")
async def get_best_model(model_name = None, watch_metric: str = "accuracy", greater_is_better: bool = True):
    best_model_info = model_manager._get_best_model_info(model_name, watch_metric, greater_is_better)
    
    if best_model_info is None:
        if model_name is None:
            exception_detail = "No models were trained yet."
        else:
            exception_detail = f"Model: '{model_name}' not found."
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exception_detail)
    
    return {
            "model_name": best_model_info["model_name"],
            "metric": watch_metric,
            "best_metric": best_model_info["best_metric"],
        }


# start training (train_config)
@router.post("/train_model", name="Start training process")
async def train_model(model_name: str, train_config: list = None):
    train_config = TrainConfig(model_name=model_name)

    coin_data = CoinData()

    training = Training(coin_data, train_config)

    run_output = training.run() 

    return {"run_history": {run_output}}




