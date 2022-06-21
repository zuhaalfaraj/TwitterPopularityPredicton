from project.data.s3_connection import S3Connection
from project.data.preprocess_data import Preprocess
from project.training.model import BERTRegressor
from transformers import BertTokenizer
from config import config
import torch


class TweetModel:
    def __init__(self, text, lang):
        self.text= text
        self.lang = lang
        self.preprocessor = Preprocess
        self.tokenizer = BertTokenizer.from_pretrained(
            config['model_checkpoint'], do_lower_case=True, do_basic_tokenize=True,
            never_split=None)
        self.model = BERTRegressor()

    def read_model(self):
        self.model = S3Connection().read_model(config['s3_model_dir'], self.model)

    def get_assessment(self):
        text = self.preprocessor.clean(self.text,self.lang)
        input_record = self.tokenizer(text, **config['tokenizer_param'])
        input_record = {key: torch.tensor(val).reshape(1,-1) for key, val in input_record.items()}
        self.read_model()
        output = self.model(**input_record)

        return output







