import time
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, LongformerModel, LongformerTokenizer
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# 解决权重无用警告
from transformers import logging

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 20211206
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# device = torch.device("mps")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = "./train.csv"
df = pd.read_csv(data_path)
# df = df.drop_duplicates(subset=['label', 'review'], keep='first')
# df = df.drop_duplicates(subset=['review'], keep=False).reset_index(drop=True)
df = df.sample(30000)
PRE_TRAINED_MODEL_NAME = 'allenai/longformer-base-4096'
tokenizer = LongformerTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
longformer_model = LongformerModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
token_lens = []
for txt in df.review:
    tokens = tokenizer.encode(txt, max_length=600)
    token_lens.append(len(tokens))

class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.reviews)
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,  #使用 longformer 可以支持到4096tokens
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )
        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }

df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df.review.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

MAX_LEN = 600
BATCH_SIZE = 8
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

logging.set_verbosity_error()
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.longformer = LongformerModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.longformer.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(output['pooler_output'])
        return self.out(output)

model = SentimentClassifier(n_classes=25)
model = model.to(device)

EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        start = time.time()
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print(f'Train loss {train_loss} accuracy {train_acc} time {time.time() - start}')
        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(df_test)
        )
        print(f'TEST  loss {test_loss} accuracy {test_acc} time {time.time() - start}')
        print()
