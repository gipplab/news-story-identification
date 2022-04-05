from configparser import ParsingError
from ECB import ECBDocument, Event, Mention, MENTION_ACTIONS, MENTION_HUMANS, MENTION_LOCATIONS, MENTION_NON_HUMANS, MENTION_TIMES
import uuid
import json

def build_giveme5w1h_training_dataset(document: ECBDocument, outputFolder="data"):
    id = uuid.uuid1().hex
    data = {
        'category': 'uncategorized',
        'dId': id,
        'date_publish': "1970-01-01 00:00:00",
        'description': "",
        'filename': id,
        'fiveWoneH': {},
        'mimeType': "text/xml",
        'originalFilename' : document.doc_name,
        'parsingError': "true",
        'publisher': "Unknown publisher",
        'text': ' '.join(document.text),
        'title': "Unknown title"
    }
    

    events = document.get_events()

    fiveWoneH = {
        "how": {
            "annotated": [],
            "label": "how"
        },
        "why": {
            "annotated": [],
            "label": "why"
        },
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
    #howAnnotations = []     # Unused for the thesis
    #whyAnnotations = []     # Unused for the thesis

    # TODO find out how coderPhraseCount is evaluated by the Golden Standard dataset
    defaultCoderPhraseCount = 1     

    for event in events:
        for mention in event.mentions:
            annotation = {
                "coderPhraseCount" : defaultCoderPhraseCount,
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

    json_string = json.dumps(data, indent=4)
    # print(json_string)

    with open(f'{outputFolder}/{id}.json', 'w') as file:
        file.write(json_string)
        print(f'Document {document.doc_name} converted to ECB+ format in file {outputFolder}/{id}.json')