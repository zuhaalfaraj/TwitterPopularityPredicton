from project.data.s3_connection import S3Connection
from project.data.preprocess_data import Preprocess
from project.training.model import BERTRegressor
from transformers import BertTokenizer
from config import config
import torch


class TweetModel:
    def __init__(self):
        self.preprocessor = Preprocess
        self.tokenizer = BertTokenizer.from_pretrained(
            config['model_checkpoint'], do_lower_case=True, do_basic_tokenize=True,
            never_split=None)

    def get_assessment(self, text, lang, model):
        text = self.preprocessor.clean(text,lang)
        input_record = self.tokenizer(text, **config['tokenizer_param'])
        input_record = {key: torch.tensor(val).reshape(1,-1) for key, val in input_record.items()}
        output = model(**input_record)
        return output







