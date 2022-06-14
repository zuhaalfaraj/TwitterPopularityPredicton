import torch
import wandb
from tqdm import tqdm
import sklearn


class TrainingLoop:
    def __init__(self, model, criterion, optimizer, device, log_to_wandb=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_to_wandb = log_to_wandb

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

        loss, accuracy, f1_score, precision_score, recall_score = self.calculate_metrics(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss, accuracy, f1_score, precision_score, recall_score

    def training_loop(self, n_epoch, save_model_path, train_loader, valid_loader):
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(n_epoch):
            print('EPOCH {}:'.format(epoch_number + 1))
            avgtrain_loss = 0
            avgval_loss = 0

            running_training_loss, running_training_accuracy, running_training_f1_score, running_training_precision_score, running_training_recall_score = 0, 0, 0, 0, 0
            running_val_loss, running_val_accuracy, running_val_f1_score, running_val_precision_score, running_val_recall_score = 0, 0, 0, 0, 0

            self.model.train()
            with tqdm(train_loader, unit="training loop") as batch:
                for i, b in enumerate(batch):

                    loss, accuracy, f1_score, precision_score, recall_score = self.one_batch(b, train=True)

                    running_training_loss += loss
                    running_training_accuracy += accuracy
                    running_training_f1_score += f1_score
                    running_training_precision_score += precision_score
                    running_training_recall_score += recall_score

                    avgtrain_loss = running_training_loss / (i + 1)
                    avgtrain_accuracy = running_training_accuracy / (i + 1)
                    avgtrain_f1_score = running_training_f1_score / (i + 1)
                    avgtrain_precision_score = running_training_precision_score / (i + 1)
                    avgtrain_recall_score = running_training_recall_score / (i + 1)

                    if not i % 50 and self.log_to_wandb:
                        wandb.log({'running_train_loss': avgtrain_loss, 'avgtrain_accuracy': avgtrain_accuracy})

            if self.log_to_wandb:
                logged_metrics = {'train_loss_per_epoch': avgval_loss, 'train_accuracy': avgtrain_accuracy,
                                  'train_f1_score': avgtrain_f1_score,
                                  'train_precision_score': avgtrain_precision_score,
                                  'train_recall_score': avgtrain_recall_score}

                wandb.log(logged_metrics)

            self.model.eval()
            with tqdm(valid_loader, unit="validation loop") as batch:
                for i, b in enumerate(batch):
                    loss, accuracy, f1_score, precision_score, recall_score = self.one_batch(b, train=False)

                    running_val_loss += loss
                    running_val_accuracy += accuracy
                    running_val_f1_score += f1_score
                    running_val_precision_score += precision_score
                    running_val_recall_score += recall_score

                    avgval_loss = running_training_loss / (i + 1)
                    avgval_accuracy = running_training_accuracy / (i + 1)
                    avgval_f1_score = running_training_f1_score / (i + 1)
                    avgval_precision_score = running_training_precision_score / (i + 1)
                    avgval_recall_score = running_training_recall_score / (i + 1)

            if self.log_to_wandb:
                logged_metrics = {'val_loss_per_epoch': avgval_loss, 'val_accuracy': avgval_accuracy,
                                  'val_f1_score': avgval_f1_score, 'val_precision_score': avgval_precision_score,
                                  'val_recall_score': avgval_recall_score}
                wandb.log(logged_metrics)

            print('LOSS train {} valid {}'.format(avgtrain_loss, avgval_loss))
            print('ACCURACY train {} valid {}'.format(avgtrain_accuracy, avgval_accuracy))
            print('F1 SCORE train {} valid {}'.format(avgtrain_f1_score, avgval_f1_score))
            print('PRECISION train {} valid {}'.format(avgtrain_precision_score, avgval_precision_score))
            print('RECALL train {} valid {}'.format(avgtrain_recall_score, avgval_recall_score))

            # Track best performance, and save the model's state
            if avgval_loss < best_vloss:
                best_vloss = avgval_loss
                torch.save(self.model.state_dict(), save_model_path)

            epoch_number += 1

        self.model.load_state_dict(torch.load(save_model_path))

        return self.model

    def calculate_metrics(self, model_output, y):
        loss = self.criterion(model_output, y)
        predictions = model_output.data.cpu().argmax(dim=1)
        y = y.cpu()
        accuracy = sklearn.metrics.accuracy_score(y, predictions)
        f1_score = sklearn.metrics.f1_score(y, predictions, average='macro')
        precision_score = sklearn.metrics.precision_score(y, predictions, average='macro')
        recall_score = sklearn.metrics.recall_score(y, predictions, average='macro')

        return loss, accuracy, f1_score, precision_score, recall_score


