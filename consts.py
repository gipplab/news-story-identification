dataset = {
    'QUORA_QA_DUPLICATES': {
        'Full': {    
            'FOLDER': './data/quora_qa_pairs',
            'FILES':  ['quora_duplicate_questions.tsv']
        }
    },
    'POLUSA': {
        '6k': {    
            'FOLDER': './data/polusa_polarity_balanced_6k',
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
            'FOLDER': './data/polusa_polarity_balanced_432k',
            'FILES': ['data.csv']
        },
        'Full': {    
            'FOLDER': './data/polusa/polusa_balanced',
            'FILES':  ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
        },
        'sentences_6k': {
            'FOLDER': './data/polusa_sentences_6k',
            'FILES':  ['data.csv']
        },
        'sentences_15k': {
            'FOLDER': './data/polusa_sentences_15k',
            'FILES':  ['data.csv']
        },
        'sentences_30k': {
            'FOLDER': './data/polusa_sentences_30k',
            'FILES':  ['data.csv']
        }
    },
    'GROUNDNEWS': {
        'Full': {
            'FOLDER': './data/ground.news',
            'FILES': ['./0/data.csv', './1/data.csv', './2/data.csv', './3/data.csv', './4/data.csv', './5/data.csv']
        }
    }
}

openShell = True
visualize = True
inference = True
# output folders
baseFolders = 'bias_results'
by_source_selection = 'source_selection'
by_commission = 'commission'
by_omission = 'omission'

# paraphrase_identifier_modelname = 'sentence-transformers/paraphrase-albert-small-v2' 
# # experimented {'eval_loss': 0.9908375144004822, 'eval_accuracy': 0.51630615640599, 'eval_precision': 0.5075029228581543, 'eval_recall': 0.5146362053182117, 'eval_f1': 0.5062493480514966, 'eval_runtime': 775.5918, 'eval_samples_per_second': 7.749, 'eval_steps_per_second': 0.485, 'epoch': 2.0}
# paraphrase_identifier_modelname = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
# paraphrase_identifier_modelname = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
paraphrase_identifier_modelname = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
# polarity_classifier_path = 'checkpoint-348'
# polarity_classifier_path = 'checkpoint-2569'
# polarity_classifier_path = 'checkpoint-3375-bak' # <- reasonable
polarity_classifier_path = 'checkpoint-1130-paraphrase-albert-small-v2'  # <- sentences 15k
# polarity_classifier_path = 'checkpoint-174'
# polarity_classifier_path = 'checkpoint-348'
# polarity_classifier_path = 'checkpoint-5138-bak'
# polarity_classifier_path = 'checkpoint-11250'

Labels = ['LEFT', 'CENTER', 'RIGHT']

Doc2Vec_Polarity_classifier_filepath = "./model/polarity_classifier_doc2vec.model"
paraphrase_threshold = 0.65
reused_threshold = 0.8

COLOR_CODE = {
    'LEFT': 'red',
    'CENTER': 'gray',
    'RIGHT': 'blue'
}

# Polarity classification training configurations
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
