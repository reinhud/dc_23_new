from src.data.datasets.coin_data import CoinData
from src.training.config.train_config import TrainConfig
from src.training.training import Training

if __name__ == '__main__':
    train_config = TrainConfig()

    coin_data = CoinData()

    training = Training(coin_data, train_config)

    training.run()
