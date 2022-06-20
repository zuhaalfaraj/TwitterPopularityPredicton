import torch
import wandb
from tqdm import tqdm
from ..data.s3_connection import S3Connection
from config import config

class TrainingLoop:
    def __init__(self, model, criterion, optimizer, device, metrics=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics

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

        return loss, outputs

    def batch_loop(self, batch, train=True):
        running_loss, avg_loss = 0, 0
        for i, b in enumerate(batch):
            loss, _ = self.one_batch(b, train=train)
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
                S3Connection().upload_model(save_model_path, self.model)

            epoch_number += 1

        self.model = S3Connection().read_model(save_model_path, self.model)

        return self.model


class WandbTrainingLoop(TrainingLoop):
    def __init__(self, model, criterion, optimizer, device, metrics=None):
        super().__init__(model, criterion, optimizer, device)

        run = wandb.init(project="twitter_viral")

        wandb.config['TRAIN_BATCH_SIZE'] = config['TRAIN_BATCH_SIZE']
        wandb.config['VALID_BATCH_SIZE'] = config['VALID_BATCH_SIZE']
        wandb.config['TEST_BATCH_SIZE'] = config['TEST_BATCH_SIZE']
        wandb.config['device'] = device
        wandb.config['model_checkpoint'] = config['model_checkpoint']
        wandb.config['tokenizer_param'] = config['tokenizer_param']
        wandb.config['splitting_ratio'] = config['splitting_ratio']
        wandb.config['random_state'] = config['random_state']
        wandb.config['classes_num'] = config['classes_num']
        wandb.config['learning_rate'] = config['learning_rate']

    def batch_loop(self, batch, train=True):
        reporting_i = 100
        running_loss, avg_loss = 0, 0
        for i, b in enumerate(batch):
            loss, outputs = self.one_batch(b, train=train)

            running_loss += loss
            avg_loss = running_loss / (i + 1)

            if not i % reporting_i:
                if train:
                    logs = {'running_loss_train': avg_loss}
                else:
                    logs = {'running_loss_val': avg_loss}

                wandb.log(logs)

        if train:
            logs = {'EPOCH_loss_train': avg_loss}
        else:
            logs = {'EPOCH_loss_val': avg_loss}
        wandb.log(logs)

        return avg_loss



