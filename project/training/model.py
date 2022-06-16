from transformers import BertForSequenceClassification


class BertModel(BertForSequenceClassification):
    def forward(self, **kwargs):
        return super().forward(**kwargs).logits