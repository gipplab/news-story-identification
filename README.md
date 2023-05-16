# Towards identification of bias by source selection, commission and omission in a collection of news articles by identification of reused text fragments

This repository holds the source code of the master thesis title above.

![alt text](https://github.com/gipplab/news-story-identification/github.jpg?raw=true)

# Step 0: Prerequisites

`Python 3.7.0+`

`Node.js v18.12.1`

`npm v8.19.2`

# Step 1: Dataset preparation

Dataset is structured collection of news articles and grouped by a general topic in a CSV file. The file must have following fields:

| Field       | Description                                       | Example                                                                                                                                                                                                                             |
| ----------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| id          | Integer                                           | 1                                                                                                                                                                                                                                   |
| title       | Article's headline                                | Russia Demanded Spy Serving Life Sentence In Germany Swapped In Exchange For Paul Whelan                                                                                                                                            |
| text        | Article's main content                            | A report by CNN says that Russia demanded that Russian spy, Vadim Krasikov, be released along with Viktor Bout, a Russian arms dealer who was serving a 25-year sentence in the US, in exchange for Paul Whelan and Brittney Griner |
| outlet_name | Outlet name (optional)                            | CNN                                                                                                                                                                                                                                 |
| datetime    | Published date with format dd/MM/yyyy hh:mm       | 9/12/2022 11:12                                                                                                                                                                                                                     |
| label       | Polarity assigned by outlet (LEFT, CENTER, RIGHT) | CENTER                                                                                                                                                                                                                              |

# Step 2: Analysis of bias by COSS

Clone the project

```
$ git clone https://github.com/gipplab/news-story-identification
```

Install dependencies

```
$ pip install -r requirements.txt
```

Put your dataset in `./data` folder, create more folders for your dataset if needed

Open the file `./consts.py`, find the variable `dataset` and create a new dictionary object to specify your data files and directory path, a dataset can have multiple versions if needed. Example:

```
dataset = {
    'POLUSA': {
        '300k': {
            'FOLDER': './data/polusa_300k',
            'FILES': ['data.csv']
        },
        'Full': {
            'FOLDER': './data/polusa/polusa_balanced',
            'FILES':  ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
        },
    }
}
```

Open the file `./by_coss`, find the variables `DATASET` and `DATASET_VERSION` and update them with your dataset name and version of choice. Example:

```
DATASET = 'POLUSA'
DATASET_VERSION = 'Full'
```

Run following command

```
$ python ./by_coss.py
```

Results are generated in the folder `./data/<your_dataset_path>/by_coss`.

# Step 3: Visualization

Go to the folder `vis`.

```
$ cd vis
```

Install dependencies

```
$ ./vis> npm install
```

Copy your analysis results from the folder `./data/<your_dataset_path>/by_coss` to `./public/`

Open the file `./environments.js`, find the variable `dataFolder` and update with your new path which you've just copied your results to (keep the keyword `exports`). Example

```
exports.dataFolder = `./public/data/ground.news/by_coss/`;
```

Run the following command

```
$ ./vis> node index.js
```

Open your favorite browser and visit `http://localhost/` and start exploring results.

# Other configurables

## Change transformers models

Path to the paraphrase identification and polarity classification models can be edited by opening the file `./consts.py`, find and edit the path for these two variables: `paraphrase_identifier_modelname` and `polarity_classifier_path`. The path can be a name to a huggingface model or in local folder.

Example:

```
# Hugging Face model
paraphrase_identifier_modelname = 'sentence-transformers/paraphrase-MiniLM-L3-v2'

# Local folder
polarity_classifier_path = './checkpoint-1130-paraphrase-albert-small-v2' #
```

## Fine tune Polarity classfication model

The polarity classification model is fined-tune by either changing the pre-trained model in the variable `polarity_classifier_path` or edit training parameters, which is an object variable `polarity_classification_configurations` in the `./consts.py` file.
Example:

```
polarity_classification_configurations = {
    "modelname": paraphrase_identifier_modelname,
    "output_dir": "model",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 2,
    "weight_decay": 0.01,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "push_to_hub": False,
}
```

Explanations of the parameters: https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/trainer#transformers.TrainingArguments
