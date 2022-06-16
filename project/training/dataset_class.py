import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, tokenizer_param, preprocessing=None, get_text=False):
        self.text = list(text['text_processed'])
        self.labels = list(text['target'])
        self.get_text = get_text

        self.encodings = tokenizer(self.text, **tokenizer_param)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["label"] = torch.tensor(int(self.labels[idx]))
        if self.get_text:
            item['text'] = self.text[idx]

        return item

    def __len__(self):
        return len(self.labels)