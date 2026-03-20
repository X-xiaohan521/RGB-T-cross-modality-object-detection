import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

class SM3Det(nn.Module):
    def __init__(self):
        super(SM3Det, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bert(x)
        return x