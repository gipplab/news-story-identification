import pandas as pd
import numpy as np
import evaluate
import code
import datasets
from os.path import join
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline

def read_data(files):
    li = []
    for filename in files:
        print(f'=== Reading {filename}')
        df = pd.read_csv(join(DATA_FOLDER, filename), index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def preprocess_function(data):
    return tokenizer(data["text"], truncation=True, padding=True)
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

if __name__ == "__main__":
    trainning = True
    if trainning == True:
        # Polusa 6k
        # DATA_FOLDER = './data/polusa_polarity_balanced_6k'
        # files = ['data.csv']

        # Polusa 90k
        DATA_FOLDER = './data/polusa_polarity_balanced_90k'
        files = ['data.csv']

        # Polusa Full
        # DATA_FOLDER = './data/polusa/polusa_balanced'
        # files = ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']

        df = read_data(files)

        #  Remove UNDEFINED polarity        
        df = df.drop(df[df.label == 'UNDEFINED'].index)

        #  Remove other columns (keep only 'text' and 'label')
        df = df.drop(['id', 'date_publish', 'outlet', 'headline', 'lead', 'authors', 'domain', 'url'], axis=1)

        #  Convert polarity leaning from str to int
        df.label[df.label=='LEFT'] = 0
        df.label[df.label=='CENTER'] = 1
        df.label[df.label=='RIGHT'] = 2
        print(f'LEFT articles: {len(df[df.label == 0])}\nCENTER articles: {len(df[df.label == 1])}\nRIGHT articles: {len(df[df.label == 2])}')

        df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.6 * len(df)), int(.8 * len(df))])
        
        print(f'Train size: {len(df_train)}\nValidation size: {len(df_val)}\nTest size: {len(df_test)}')

        ds_train = datasets.Dataset.from_pandas(df_train)
        ds_val = datasets.Dataset.from_pandas(df_val)
        ds_test = datasets.Dataset.from_pandas(df_test)

        ds_train_tokenized = ds_train.map(preprocess_function, batched=True)
        ds_val_tokenized = ds_val.map(preprocess_function, batched=True)
        ds_test_tokenized = ds_test.map(preprocess_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        accuracy = evaluate.load("accuracy")

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
            eval_dataset=ds_test_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        code.interact(local=locals())
    else:
        from transformers import pipeline

        # This text's real truth is 'RIGHT'
        text = '''A cargo plane crashed Monday in a residential area just outside the main airport in Kyrgyzstan, killing at least 37 people, the Emergency Situations Ministry said on Monday.
                The Turkish Boeing 747 crashed just outside the Manas airport, south of the capital Bishkek, killing people in the residential area adjacent to the airport as well as those on the plane.
                Reports of the death toll on Monday ranged from 37 people according to emergency officials in the Central Asian nation, to 31 reported by the presidential press office which also said rescue teams had recovered parts of nine bodies. Fifteen people including six children have been hospitalized.
                Images from the scene showed the plane's nose stuck inside a brick house and large chunks of debris scattered around.
                Several dozen private houses cluster just outside the metal fence separating the cottages from the runway. Manas has been considerably expanded since the United States began to operate a military installation at the Manas airport, using it primarily for its operations in Afghanistan. American troops vacated the base and handed it over to the Kyrgyz military in 2014.
                ""I woke up because of a bright red light outside,"" Baktygul Kurbatova, who was slightly injured, told local television. ""I couldn't understand what was happening. It turns out the ceiling and the walls were crashing on us. I was so scared but I managed to cover my son's face with my hands so that debris would not fall on him.""
                More than a thousand rescue workers were at the scene by late morning in the residential area where 15 houses were destroyed, Deputy Prime Minister Mukhammetkaly Abulgaziyev said.
                The cause of the crash was not immediately clear. Kyrgyz Emergency Situations Minister Kubatbek Boronov told reporters that it was foggy at Manas when the plane came down but weather conditions were not critical. The plane's flight recorders have not yet been found.
                The plane, which had departed from Hong Kong, belonged to Istanbul-based cargo company ACT Airlines. It said in an emailed statement that the cause was unknown.
                Turkish Foreign Minister Mevlut Cavusoglu on Monday called his Kyrgyz counterpart, Erlan Abdildaev, to offer Turkey's condolences, the Turkish Foreign Ministry said.'''

        print(f'Inferring {text[:50]}...')

        classifier = pipeline("polarity-classification", model="./model/checkpoint-450")

        print(classifier(text))
        
        code.interact(local=locals())