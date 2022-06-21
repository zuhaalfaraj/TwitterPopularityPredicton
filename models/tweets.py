from project.data.s3_connection import S3Connection
from project.data.preprocess_data import Preprocess
from project.training.dataset_class import TextDataset
from project.training.model import BertModel
from transformers import BertTokenizer
from config import config
import torch


class Tweet:
    def __int__(self):
        self.preprocessor = Preprocess
        self.tokenizer = BertTokenizer.from_pretrained(
            config['model_checkpoint'], do_lower_case=True, do_basic_tokenize=True,
            never_split=None)
        self.model = BertModel()


    def read_model(self):
        self.model = S3Connection().read_model(config['s3_model_dir'], self.model)

    def get_assessment(self, text, lang):
        text = self.preprocessor(text,lang)
        input_record = self.tokenizer(text, **config['tokenizer_param'])
        input_record = {key: torch.tensor(val).reshape(1,-1) for key, val in input_record.items()}
        output = self.model(**input_record)

        return output







