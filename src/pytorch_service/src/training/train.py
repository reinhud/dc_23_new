from pytorch_service.src.data.datasets.coin_data import CoinData
from pytorch_service.src.training.config.train_config import TrainConfig
from pytorch_service.src.training.training import Training

if __name__ == "__main__":
    train_config = TrainConfig()

    coin_data = CoinData()

    training = Training(coin_data, train_config)

    run_history = training.run() 

    print(run_history._metrics)


