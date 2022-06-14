import os
import sys
import json
import argparse
from datetime import datetime
from ECB import ECBDocument
from utils import build_giveme5w1h_training_dataset

###   Convert ECB+ dataset into Giveme5W1H-XML format for training
def ecbplus_conversion():
    dataPath = 'data/ECB+'
    folders = os.listdir(dataPath)
    docs = []
    convert_to_goldenstandard_format = True
    for subFolder in folders:
        if os.path.isdir(f'{dataPath}/{subFolder}'):
            items = os.listdir(f'{dataPath}/{subFolder}')
            for file in items:
                # print(f'./{dataPath}/{subFolder}/{file}')

                doc = ECBDocument()
                doc.read(f'./{dataPath}/{subFolder}/{file}')
                docs.append(doc)

                if convert_to_goldenstandard_format:
                    build_giveme5w1h_training_dataset(doc, outputFolder="data/ECBplus_giveme5w1h")
                
    print(f'{len(docs)} documents read')

##  4W-Questions Extraction Pipeline
def process(input_dir, output_dir):
    pass

def load_best_weights(path):
    with open(path, encoding='utf-8') as data_file:
        data = json.load(data_file)
    return data['best_dist']['weights']
    
def load_weights(weights_dir=""):
    weights = {
        "who": [],
        "what": [],
        "where": [],
        "when": []
    }

    if len(weights_dir) == 0:
        return weights
    


    return weights

if __name__ == "__main__":
    if len(sys.argv) == 3:
        beginning = datetime.now()

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        if output_dir[-1] != "/":
            output_dir+="/"

        # Instantiate the parser
        parser = argparse.ArgumentParser(description='Optional app description')

        # Optional argument
        parser.add_argument('--weights', type=str, nargs='?', const=f'', default='',
                            help='Directory points to weights folder')

        args = parser.parse_args()
        weights = load_weights(args.weights)

        print(f"=== DONE ! Total times for Task 2 is {datetime.now() - beginning}")
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: python ./task2.py {input-dir} {output-dir}",
                         "Optional: --weights {weight-dir}"]))