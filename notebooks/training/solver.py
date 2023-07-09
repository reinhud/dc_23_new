"""Implement a solver for the training process."""
import torch
import torch.nn as nn
from tqdm.notebook import tnrange
from training.training_history.training_history import TrainingHistory


class ClassificationSolver:
    def __init__(
        self,
        model: nn.Module,
        dataloader_train,
        dataloader_test,
        loss_fn,
        optimizer,
        scheduler=None,
    ):
        # check and use gpu if available
        if torch.cuda.is_available():
            self.device = "cuda"
            print(
                "Perfect, you're putting your GPU to work and using CUDA for your computations."
            )
        else:
            self.device = "cpu"
            print(
                "What a bummer, your using your CPU. Be careful when trying to train the model, will be slow as hell."
            )

        self.model = model.to(self.device)  # move the model device
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        # set up history for tracking model
        self.history = TrainingHistory(
            model_name=model.__class__.__name__,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def train_step(self):
        # Put model in train mode
        self.model.train()

        train_loss, train_acc = 0, 0

        # Iterate over the training data
        for batch, (X, y) in enumerate(self.dataloader_train):
            # Send data to device
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_pred = self.model(X)

            # Calculate and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # Optimizer zero grad
            self.optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            self.optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(self.dataloader_train)
        train_acc = train_acc / len(self.dataloader_train)

        return train_loss, train_acc

    def test_step(self):
        # Put model in eval mode
        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(self.dataloader_test):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                test_pred_logits = self.model(X)

                # Calculate and accumulate loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(self.dataloader_test)
        test_acc = test_acc / len(self.dataloader_test)

        return test_loss, test_acc

    def train(self, epochs=5):
        print("#" * 80)
        print("Training started...")
        print("#" * 80)
        for epoch in tnrange(epochs, desc="Epochs"):
            # Train step
            train_loss, train_acc = self.train_step()

            # Test step
            test_loss, test_acc = self.test_step()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Update training history
            if test_acc > max(self.history.test_acc, default=0):
                self.history.best_model_params = self.model.state_dict().copy()
            self.history.train_loss.append(train_loss)
            self.history.train_acc.append(train_acc)
            self.history.test_loss.append(test_loss)
            self.history.test_acc.append(test_acc)

            # Print epoch summary
            print(f"Epoch: {epoch + 1}")
            print("-" * 40)
            print(f"Train loss: {train_loss:.4f}")
            print(f"Train acc: {train_acc:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test acc: {test_acc:.4f}\n")

        return self.history
