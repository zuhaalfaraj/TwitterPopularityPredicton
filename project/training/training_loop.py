import torch
import wandb
from tqdm import tqdm
from ..data.s3_connection import S3Connection


class TrainingLoop:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def one_batch(self, batch, train=True):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        y = batch['label']
        del batch['label']
        x = batch

        if not train:
            with torch.no_grad():
                outputs = self.model(**x)

        else:
            self.optimizer.zero_grad()
            outputs = self.model(**x)
        loss = self.criterion(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss

    def batch_loop(self, batch):
        running_loss, avg_loss = 0, 0
        for i, b in enumerate(batch):
            loss = self.one_batch(b, train=True)
            running_loss += loss
            avg_loss = running_loss / (i + 1)

        return avg_loss

    def training_loop(self, n_epoch, save_model_path, train_loader, valid_loader):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch_number + 1))

            self.model.train()
            with tqdm(train_loader, unit="training loop") as batch:
                avgtrain_loss = self.batch_loop(batch)

            self.model.eval()
            with tqdm(valid_loader, unit="validation loop") as batch:
                avgval_loss = self.batch_loop(batch)

            print('LOSS train {} valid {}'.format(avgtrain_loss, avgval_loss))

            # Track best performance, and save the model's state
            if avgval_loss < best_vloss:
                best_vloss = avgval_loss
                S3Connection.upload_model(save_model_path, self.model)

            epoch_number += 1

        self.model = S3Connection.read_model(save_model_path, self.model)

        return self.model




