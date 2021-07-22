import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm
from loguru import logger

# device = torch.device("cuda") # gpu 사용

tokenizer = AutoTokenizer.from_pretrained(
    "monologg/koelectra-small-v2-discriminator"
)


def getitem(text): 
# input text make in model input(tokenize, make attention mask)

    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        pad_to_max_length=True,
        add_special_tokens=True,
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask


def test():
    # get a pre-trained model from Transformer 
    model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-small-v2-discriminator"
    )

    try:
        # load trained model 
        model.load_state_dict(torch.load("model.pt"))
    except:
        # if not have trained model, first train in train code
        pass

    batch_size = 128
    optimizer = AdamW(model.parameters(), lr=1e-5)

    while True: # predict sentiment
        logger.info('please input reply >')
        reply = input()
        if reply == 'quit':
            break
        input_ids, attention_mask = getitem(reply)
        y_pred = model(input_ids, attention_mask=attention_mask)[0]
        temp, predicted = torch.max(y_pred, 1)

        logger.info(predicted)
        if predicted == 0:
            logger.info('Negative')
        else:
            logger.info('Positive')
