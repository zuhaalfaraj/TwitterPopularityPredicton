from ..data.s3_connection import S3Connection
import torch
from ..data.split_data import SplitData
from .dataset_class import TextDataset
from torch.utils.data import Dataset, DataLoader
from .model import BertModel
from transformers import AdamW
from training_loop import TrainingLoop

class TrainingPipeline:
    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        # 1. Read the dataset from S3
        data = S3Connection().read_data('preprocessed_data/twitter_data.csv')  # the dir should be config
        # 2. Split the dataset
        splitter = SplitData(0.2, 101)
        text_train, text_test, text_val = splitter(data.dropna())
        # 3. Define the dataset class
        train_dataset = TextDataset(text=text_train, tokenizer_param=tokenizer_param, tokenizer=tokenizer)
        test_dataset = TextDataset(text=text_test, tokenizer_param=tokenizer_param, tokenizer=tokenizer)
        val_dataset = TextDataset(text=text_val, tokenizer_param=tokenizer_param, tokenizer=tokenizer)
        # 4. Define the dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
        # 5. Define the model
        self.model = BertModel.from_pretrained(
            model_checkpoint,
            num_labels=classes_num)
        self.model.to(self.device)

        # 6. Define the training criterion & optimizer.
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss(weight=classes_weight)

        # 7. Define the traininer
        self.trainer = TrainingLoop(self.model, self.criterion, self.optimizer, device=self.device)

    def start(self, n_epoch):
        self.trainer.training_loop(n_epoch=n_epoch, save_model_path=self.model_path, train_loader=self.train_loader,
                                   valid_loader=self.valid_loader)

    def get_model(self):
        self.model = S3Connection.read_model(self.model_path, self.model)
        return self.model