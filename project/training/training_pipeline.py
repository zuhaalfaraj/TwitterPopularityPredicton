from ..data.s3_connection import S3Connection
import torch
from ..data.split_data import SplitData
from .dataset_class import TextDataset
from torch.utils.data import Dataset, DataLoader
from .model import BERTRegressor
from transformers import AdamW, BertTokenizer
from training_loop import TrainingLoop
from config import config


class TrainingPipeline:
    def __init__(self, model_path):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        # 1. Read the dataset from S3
        data = S3Connection().read_data('preprocessed_data/twitter_data.csv')  # the dir should be config
        # 2. Split the dataset
        splitter = SplitData(0.2, 101)
        text_train, text_test, text_val = splitter(data.dropna())
            # Data length
        cls_0_len = len(text_train[text_train['target'] == 0])
        cls_1_len = len(text_train[text_train['target'] == 0])
        cls_2_len = len(text_train[text_train['target'] == 0])
        data_len = len(text_train)
        # 3. Define the dataset class
        tokenizer = BertTokenizer.from_pretrained(
            config['model_checkpoint'], do_lower_case=True, do_basic_tokenize=True, never_split=None
        )
        train_dataset = TextDataset(text=text_train, tokenizer_param=config['tokenizer_param'] , tokenizer=tokenizer)
        test_dataset = TextDataset(text=text_test, tokenizer_param=config['tokenizer_param'] , tokenizer=tokenizer)
        val_dataset = TextDataset(text=text_val, tokenizer_param=config['tokenizer_param'] , tokenizer=tokenizer)
        # 4. Define the dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=True)
        self.valid_loader = DataLoader(val_dataset, batch_size=config['VALID_BATCH_SIZE'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config['TEST_BATCH_SIZE'], shuffle=True)
        # 5. Define the model
        self.model = BERTRegressor()
        self.model.to(self.device)


        # 6. Define the training criterion & optimizer.
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'] )
        classes_weight = [cls_0_len/data_len, cls_1_len/data_len, cls_2_len/data_len]
        classes_weight = torch.tensor(classes_weight).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=classes_weight)

        # 7. Define the traininer
        self.trainer = TrainingLoop(self.model, self.criterion, self.optimizer, device=self.device)

    def start(self, n_epoch):
        self.trainer.training_loop(n_epoch=n_epoch, save_model_path=self.model_path, train_loader=self.train_loader,
                                   valid_loader=self.valid_loader)

    def get_model(self):
        self.model = S3Connection.read_model(self.model_path, self.model)
        return self.model