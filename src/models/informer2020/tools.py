import numpy as np
import torch


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data):
        if torch.is_tensor(data):
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
            std = torch.from_numpy(self.std).type_as(data).to(data.device)
            return (data - mean) / std
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if torch.is_tensor(data):
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device)
            std = torch.from_numpy(self.std).type_as(data).to(data.device)
        else:
            mean = self.mean
            std = self.std
        if getattr(data, "shape", None) is not None and data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "checkpoint.pth")
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj="type1"):
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    else:
        return

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")
