"""Define the format of the training history."""
import pickle
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns


class TrainingHistory:
    """Specifying the history saved during training."""

    HISTORY_PATH = (
        "/workspaces/data-challenge-sose23/src/training/training_history/history/"
    )

    def __init__(
        self, model_name=None, loss_fn=None, optimizer=None, scheduler=None, epochs=None
    ):
        self.model_name = model_name
        self.timestamp = datetime.now().timestamp()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_params = {}
        self.avg_epoch_loss = []
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.epochs = len(self.train_loss)

    def save(self):
        """Saves the history in a pickle file."""
        instance_name = f"{self.model_name}" + "_" + f"{self.timestamp}"
        instance_path = self.HISTORY_PATH + instance_name

        with open(instance_path, "wb") as f:
            pickle.dump(self.__dict__, f)
            print(f"History saved to {instance_path}")

    def load_histories(cls) -> list:
        """Loads all stored training histories."""
        histories = []

        # find all stored pickled histories
        pickle_hists = glob(f"{cls.HISTORY_PATH}*.pickle")

        for p_hist in pickle_hists:
            hist = TrainingHistory()
            with open(p_hist, "rb") as f:
                hist.__dict__ = pickle.loads(f)
                histories.append(hist)

        return histories

    def show_training(self):
        """Create a plot with seaborn showing loss and accuracy over epochs."""
        epochs = range(len(self.train_loss))

        sns.set_theme(style="darkgrid")

        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Training Results")

        # plot loss
        sns.lineplot(x=epochs, y=self.train_loss, label="train", ax=ax1)
        sns.lineplot(x=epochs, y=self.test_loss, label="test", ax=ax1)
        ax1.set_title("Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        # plot the accuracy in the second subplot
        sns.lineplot(x=epochs, y=self.train_acc, ax=ax2)
        sns.lineplot(x=epochs, y=self.test_acc, ax=ax2)
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        # adjust the space between the subplots
        fig.subplots_adjust(wspace=0.3)

        # add a legend to the plot
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right")

        # show the plot
        plt.show()
