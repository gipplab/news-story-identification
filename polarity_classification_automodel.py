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
        print(f'=== Reading {join(folder, filename)}')
        df = pd.read_csv(join(folder, filename), index_col=None, header=0, encoding='ISO-8859-1')
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
    precision_mi = evaluate.load("precision").compute(predictions=predictions, references=labels, average='micro')
    recall_mi = evaluate.load("recall").compute(predictions=predictions, references=labels, average='micro')
    f1_mi = evaluate.load("f1").compute(predictions=predictions, references=labels, average='micro')

    precision_ma = evaluate.load("precision").compute(predictions=predictions, references=labels, average='macro')
    recall_ma = evaluate.load("recall").compute(predictions=predictions, references=labels, average='macro')
    f1_ma = evaluate.load("f1").compute(predictions=predictions, references=labels, average='macro')

    return {**accuracy, **precision_mi, **recall_mi, **f1_mi, **precision_ma, **recall_ma, **f1_ma}

def manual_metrics(eval_pred):
    predictions, labels = eval_pred

    accuracy = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    precision_mi = evaluate.load("precision").compute(predictions=predictions, references=labels, average='micro')
    recall_mi = evaluate.load("recall").compute(predictions=predictions, references=labels, average='micro')
    f1_mi = evaluate.load("f1").compute(predictions=predictions, references=labels, average='micro')

    precision_ma = evaluate.load("precision").compute(predictions=predictions, references=labels, average='macro')
    recall_ma = evaluate.load("recall").compute(predictions=predictions, references=labels, average='macro')
    f1_ma = evaluate.load("f1").compute(predictions=predictions, references=labels, average='macro')

    return {**accuracy, **precision_mi, **recall_mi, **f1_mi, **precision_ma, **recall_ma, **f1_ma,}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

if __name__ == "__main__":
    POLUSA = {
        '6k': {    
            'FOLDER': './data/polusa_polarity_balanced_6k',
            'FILES': ['data.csv']
        },
        '90k': {    
            'FOLDER': './data/polusa_polarity_balanced_90k',
            'FILES': ['data.csv']
        },
        '432k': {    
            'FOLDER': './data/polusa_polarity_balanced_432k',
            'FILES': ['data.csv']
        },
        'Full': {    
            'FOLDER': './data/polusa/polusa_balanced',
            'FILES':  ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
        }
    }
    POLUSA_VERSION = '90k'
    FOLDER = POLUSA[POLUSA_VERSION]['FOLDER']
    FILES = POLUSA[POLUSA_VERSION]['FILES']

    # GROUNDNEWS = {
    #     'Full': {
    #         'FOLDER': './data/ground.news',
    #         'FILES': ['0/data.csv', '1/data.csv', '2/data.csv', '3/data.csv', '4/data.csv']
    #     }
    # }
    # GROUNDNEWS_VERSION = 'Full'
    # FOLDER = GROUNDNEWS[GROUNDNEWS_VERSION]['FOLDER']
    # FILES = GROUNDNEWS[GROUNDNEWS_VERSION]['FILES']

    df = read_data(FOLDER, FILES)
    df = df.dropna()
    df = df.drop(df[df.label == 'UNDEFINED'].index, errors='ignore')
    df = df.drop(['id', 'date_publish', 'outlet', 'headline', 'lead', 'authors', 'domain', 'url'], axis=1, errors='ignore')
    df.label[df.label=='LEFT'] = 0
    df.label[df.label=='CENTER'] = 1
    df.label[df.label=='RIGHT'] = 2

    print(f'LEFT articles: {len(df[df.label == 0])}\nCENTER articles: {len(df[df.label == 1])}\nRIGHT articles: {len(df[df.label == 2])}')
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])
    print(f'Train size: {len(df_train)}\nValidation size: {len(df_val)}\nTest size: {len(df_test)}')

    ds_train_tokenized = datasets.Dataset.from_pandas(df_train).map(preprocess_function, batched=True)
    ds_eval_tokenized = datasets.Dataset.from_pandas(df_val).map(preprocess_function, batched=True)
    ds_test_tokenized = datasets.Dataset.from_pandas(df_test).map(preprocess_function, batched=True)

    trainning = False
    inference = True
    openShell = True

    if trainning:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        id2label = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}
        label2id = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}

        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id)

        training_args = TrainingArguments(
            output_dir="model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
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
            eval_dataset=ds_eval_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()    

    if inference:
        # from transformers import AutoTokenizer
        import torch

        model = AutoModelForSequenceClassification.from_pretrained("./model/checkpoint-5138")
        with torch.no_grad():
            eval = []
            pred = []
            for i, row in df_test.iterrows():
                text = row['text']
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True)

                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                eval.append(row['label'])
                pred.append(predicted_class_id)

        results = manual_metrics((eval, pred))
        print(f'Results: {results}')

        # Confusion matrix
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        eval_label = [model.config.id2label[i] for i in eval]
        pred_label = [model.config.id2label[i] for i in pred]
        cm = confusion_matrix(eval_label, pred_label)
        left_left = 0
        left_center = 0
        left_right = 0
        center_left = 0
        center_center = 0
        center_right = 0
        right_left = 0
        right_center = 0
        right_right = 0
        for i in range(len(eval_label)):
            # print(f'Truth: {eval_label[i]}, Pred: {pred_label[i]}')
            if eval_label[i] == 'LEFT' and pred_label[i] == 'LEFT':
                left_left = left_left + 1
            if eval_label[i] == 'LEFT' and pred_label[i] == 'CENTER':
                left_center = left_center + 1
            if eval_label[i] == 'LEFT' and pred_label[i] == 'RIGHT':
                left_right = left_right + 1
            if eval_label[i] == 'CENTER' and pred_label[i] == 'LEFT':
                center_left = center_left + 1
            if eval_label[i] == 'CENTER' and pred_label[i] == 'CENTER':
                center_center = center_center + 1
            if eval_label[i] == 'CENTER' and pred_label[i] == 'RIGHT':
                center_right = center_right + 1
            if eval_label[i] == 'RIGHT' and pred_label[i] == 'LEFT':
                right_left = right_left + 1
            if eval_label[i] == 'RIGHT' and pred_label[i] == 'CENTER':
                right_center = right_center + 1
            if eval_label[i] == 'RIGHT' and pred_label[i] == 'RIGHT':
                right_right = right_right + 1
        print(cm)
        print('LEFT-LEFT', left_left)
        print('LEFT-CENTER', left_center)
        print('LEFT-RIGHT', left_right)
        print('CENTER-LEFT', center_left)
        print('CENTER-CENTER', center_center)
        print('CENTER-RIGHT', center_right)
        print('RIGHT-LEFT', right_left)
        print('RIGHT-CENTER', right_center)
        print('RIGHT-RIGHT', right_right)
        cm_df = pd.DataFrame(cm, index = ['LEFT','CENTER','RIGHT'], columns = ['LEFT','CENTER','RIGHT'])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_df, annot=True, fmt='d')
        plt.title('Confusion Matrix - Polarity detection training')
        plt.ylabel('Predictions')
        plt.xlabel('Truths')
        plt.show()

    if openShell:
        code.interact(local=locals())
    