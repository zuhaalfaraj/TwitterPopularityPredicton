from transformers import BertForSequenceClassification
from transformers import BertModel
import torch.nn as nn
from config import config


class BERTRegressor(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(BERTRegressor, self).__init__()
        D_in, D_out = 768, 1

        self.bert = BertModel.from_pretrained(config['model_checkpoint'], force_download=True)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))

    def forward(self, **kwargs):
        outputs = self.bert(**kwargs)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs.squeeze(1)



