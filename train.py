import numpy
import pandas
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertweetTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW

class Dataset(torch.utils.data.Dataset):
    # https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
    def __init__(self, encodings, labels):
        self.labels = torch.tensor(labels).clone().detach()
        self.encodings = {
            key: val.clone().detach()
            for key, val in encodings.items()
        }

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

    def __len__(self): return len(self.encodings["input_ids"])


# load model
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=5)
model.train()  # tell model to enter "train" mode
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# load labeled data
labeled_data = pandas.read_csv("data/2022-05-18-climate-change-labeled-samples.csv")[['index', 'text', 'label_text']].set_index('index').sort_index()
# remove irrelevant label
labeled_data = labeled_data.loc['irrelevant or no opinion on climate change' != labeled_data['label_text']]
labeled_data = labeled_data.loc['review with JunMing' != labeled_data['label_text']]

# double check for 5-labels 
print(labeled_data['label_text'].unique())
assert 5 == len(labeled_data['label_text'].unique())
labeled_data['labels'] = labeled_data['label_text'].map({
    'Strongly agree with climate change': 0,
    'Slightly agree with climate change': 1,
    'Neutral to climate change': 2,
    'Slightly disagree with climate change': 3,
    'Strongly disagree with climate change': 4
}).to_list()

# prepare data in tokens
tokenizer = BertTokenizer.from_pretrained(model_name)
encoding = tokenizer(labeled_data['text'].to_list(), return_tensors='pt', padding=True, truncation=True, max_length=128)

# split data
train_size = int(len(labeled_data) * 0.8)
eval_size = len(labeled_data) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(dataset=Dataset(encodings=encoding, labels=labeled_data['labels'].to_numpy()), lengths=[train_size, eval_size])

# train pre-trained model
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.05,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    seed=0,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=50,
)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)
trainer.train()
trainer.save_model()
trainer.evaluate(eval_dataset=eval_dataset)

# load unlabeled data
unlabeled_data = pandas.read_csv("data/test.csv")[['index', 'text', 'label_text']].set_index('index').sort_index()

# tokenize data
encoding = tokenizer(unlabeled_data['text'].to_list(), return_tensors='pt', padding=True, truncation=True, max_length=128)
# eval
output = model(input_ids = encoding['input_ids'], attention_mask = encoding['attention_mask'], token_type_ids = encoding['token_type_ids'])
logits = output[:1]    
print(logits)