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
    },
    'GROUNDNEWS': {
        'Full': {
            'FOLDER': './data/ground.news',
            # 'FILES': ['0/data.csv']
            'FILES': ['./0/data.csv', './1/data.csv', './2/data.csv', './3/data.csv', './4/data.csv']
        }
    }
}