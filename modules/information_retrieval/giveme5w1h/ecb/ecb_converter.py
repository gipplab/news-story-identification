from os import listdir
from os.path import isfile, exists, join
from information_retrieval.giveme5w1h.ecb.ECB import ECBDocument, MENTION_ACTIONS, MENTION_HUMANS, MENTION_LOCATIONS, MENTION_NON_HUMANS, MENTION_TIMES
import uuid
import json

def convert(document: ECBDocument, outputFolder="data"):
    id = uuid.uuid1().hex    
    data = {
        # 'category': 'uncategorized',
        'dId': id,
        # 'date_publish': "1970-01-01 00:00:00",
        'description': "",
        'filename': id,
        'fiveWoneH': {},
        'mimeType': "text/xml",
        'originalFilename' : document.doc_name,
        # 'parsingError': "true",
        # 'publisher': "Unknown publisher",
        'text': ' '.join(document.text),
        # 'title': "Unknown title"
    }
    

    events = document.get_events()

    fiveWoneH = {
        "where": {
            "annotated": [],
            "label": "where"
        },
        "when": {
            "annotated": [],
            "label": "when"
        },
        "what": {
            "annotated": [],
            "label": "what"
        },
        "who": {
            "annotated": [],
            "label": "who"
        }
    }

    for event in events:
        for mention in event.mentions:
            annotation = {
                # "coderPhraseCount" : defaultCoderPhraseCount,
                "text": mention.token
            }

            # WHAT
            if mention.type in MENTION_ACTIONS:                
                fiveWoneH['what']['annotated'].append(annotation)

            # WHEN
            if mention.type in MENTION_TIMES:
                fiveWoneH['when']['annotated'].append(annotation)

            # WHERE
            if mention.type in MENTION_LOCATIONS:
                fiveWoneH['where']['annotated'].append(annotation)

            # WHO
            if mention.type in MENTION_HUMANS:
                fiveWoneH['who']['annotated'].append(annotation)

            # TODO maybe put into WHERE?
            if mention.type in MENTION_NON_HUMANS:
                pass
    data['fiveWoneH'] = fiveWoneH

    with open(f'{join(outputFolder, id)}.json', 'w') as file:
        json.dump(data, file, indent=4)
    file.close()

    return data

"""
    @params sampling: 
"""
def process(input_dir, output_dir, sampling_threshold=0.8):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.xml')]
    docs = []
    sampling = {
        "training": [],
        "test": []
    }
    for i, file in enumerate(files):
        print(f'[CONVERTING] {"{0:0.2f}".format(i / len(files) * 100)}% converting to Giveme5w1h format {file} ...')
        doc = ECBDocument()
        docs.append(doc.read(join(input_dir, file)))        

        converted = convert(doc, output_dir)

        if i / len(files) <= sampling_threshold:
            sampling['training'].append(converted['dId'] + '.json')
        else:
            sampling['test'].append(converted['dId'] + '.json')

    with open(join(output_dir, 'sampling.json'), 'w', encoding='utf-8') as f:
        json.dump(sampling, f, indent=4)
    f.close()
    print(f'{len(docs)} documents read')

"""
    Validating: check if:
        true => each document has at least one question that have data
        false => there is at least 1 document which does not have anything in its 'fiveWoneH' attribute
"""
def validate(input_dir):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith('.json')]
    exist_none = True
    for i, file in enumerate(files):
        with open(join(input_dir, file), 'r') as fp:
            if file != 'sampling.json':
                doc = json.load(fp)
                print(f'[VALIDATING] {"{0:0.2f}".format(i / len(files) * 100)}% doc {doc["originalFilename"]}: "who": {len(doc["fiveWoneH"]["who"]["annotated"])} - "what": {len(doc["fiveWoneH"]["what"]["annotated"])} - "when": {len(doc["fiveWoneH"]["when"]["annotated"])} - "where": {len(doc["fiveWoneH"]["where"]["annotated"])}')
                if len(doc["fiveWoneH"]["who"]["annotated"]) == 0 and len(doc["fiveWoneH"]["what"]["annotated"]) == 0 and len(doc["fiveWoneH"]["when"]["annotated"]) == 0 and len(doc["fiveWoneH"]["where"]["annotated"]) == 0:
                    exist_none = False
        fp.close()
    print(f'[VALIDATING] Finished, result: {exist_none}')       

if __name__ == "__main__":
    input_dir = './data/ECB+'
    output_dir = './output/ECB+'
    process(input_dir, output_dir)
    validate(output_dir)

    
