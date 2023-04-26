import code
from os.path import join

import evaluate
import numpy as np
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import datasets


def read_data(folder, files):
    li = []
    for filename in files:
        print(f'=== Reading {filename}')
        df = pd.read_csv(join(folder, filename), index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def preprocess_function(data):
    return tokenizer(
        data["text"], 
        truncation=True, 
        padding=True,
        add_special_tokens=True)
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels, average='micro')
    recall = evaluate.load("recall").compute(predictions=predictions, references=labels, average='micro')
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average='micro')

    return {**accuracy, **precision, **recall, **f1}

def manual_metrics(eval_pred):
    predictions, labels = eval_pred

    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    precision = evaluate.load("precision").compute(predictions=predictions, references=labels, average='micro')
    recall = evaluate.load("recall").compute(predictions=predictions, references=labels, average='micro')
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average='micro')

    return {**accuracy, **precision, **recall, **f1}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

if __name__ == "__main__":
    POLUSA = {
        '6k': {    
            'FOLDER': './data/polusa_balanced_6k',
            'FILES': ['data.csv']
        },
        '90k': {    
            'FOLDER': './data/polusa_balanced_90k',
            'FILES': ['data.csv']
        },
        '300k': {
            'FOLDER': './data/polusa_300k',
            'FILES': ['data.csv']
        },
        '432k': {    
            'FOLDER': './data/polusa_balanced_432k',
            'FILES': ['data.csv']
        },
        
        'Full': {    
            'FOLDER': './data/polusa/polusa_balanced',
            'FILES':  ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
        }
    }

    POLUSA_VERSION = '300k'   # <-- choose version here
    FOLDER = POLUSA[POLUSA_VERSION]['FOLDER']
    FILES = POLUSA[POLUSA_VERSION]['FILES']

    df = read_data(FOLDER, FILES)
    df = df.drop(df[df.label == 'UNDEFINED'].index)
    df = df.drop(['id', 'date_publish', 'outlet', 'headline', 'lead', 'authors', 'domain', 'url'], axis=1)
    df.label[df.label=='LEFT'] = 0
    df.label[df.label=='CENTER'] = 1
    df.label[df.label=='RIGHT'] = 2

    print(f'LEFT articles: {len(df[df.label == 0])}\nCENTER articles: {len(df[df.label == 1])}\nRIGHT articles: {len(df[df.label == 2])}')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])
    print(f'Train size: {len(df_train)}\nValidation size: {len(df_val)}\nTest size: {len(df_test)}')

    ds_train_tokenized = datasets.Dataset.from_pandas(df_train).map(preprocess_function, batched=True)
    ds_val_tokenized = datasets.Dataset.from_pandas(df_val).map(preprocess_function, batched=True)
    ds_test_tokenized = datasets.Dataset.from_pandas(df_test).map(preprocess_function, batched=True)

    trainning = True
    openShell = True

    if trainning:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        id2label = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
        label2id = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}

        # model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id)
        model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2", num_labels=3, id2label=id2label, label2id=label2id)
        # model = AutoModelForSequenceClassification.from_pretrained("./model/checkpoint-5138", num_labels=3, id2label=id2label, label2id=label2id)
        training_args = TrainingArguments(
            output_dir="model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train_tokenized,
            eval_dataset=ds_val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()    

    if openShell:
        code.interact(local=locals())
    