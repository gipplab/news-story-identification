# Towards identification of bias by source selection, commission and omission in a collection of news articles by identification of reused text fragments

This repository holds the source code of the master thesis title above.

# Step 0: Prerequisites:

`Python 3.7.0+`

`Node.js v18.12.1`

`npm v8.19.2`

# Step 1: Dataset preparation

Dataset is structured collection of news articles and grouped by a general topic in a CSV file. The file must have following fields:

| Field       | Description                                       | Example                                                                                                                                                                                                                             |     |     |     |     |     |     |     |
| ----------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| id          | Integer                                           | 1                                                                                                                                                                                                                                   |     |     |     |     |     |     |     |
| title       | Article's headline                                | Russia Demanded Spy Serving Life Sentence In Germany Swapped In Exchange For Paul Whelan                                                                                                                                            |     |     |     |     |     |     |     |
| text        | Article's main content                            | A report by CNN says that Russia demanded that Russian spy, Vadim Krasikov, be released along with Viktor Bout, a Russian arms dealer who was serving a 25-year sentence in the US, in exchange for Paul Whelan and Brittney Griner |     |     |     |     |     |     |     |
| outlet_name | Outlet name (optional)                            | CNN                                                                                                                                                                                                                                 |     |     |     |     |     |     |     |
| datetime    | Published date with format dd/MM/yyyy hh:mm       | 9/12/2022 11:12                                                                                                                                                                                                                     |     |     |     |     |     |     |     |
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

Run the following command and start viewing results

```
$ ./vis> node index.js
```

# Other configurables
