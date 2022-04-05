from torch import true_divide
from ECB import ECBDocument
import os
from utils import build_giveme5w1h_training_dataset

#   Read all ECB+ documents
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

# for doc in docs:
#     if doc.doc_id == "DOC15653231215902085" or doc.doc_id == "DOC15645929111211951##":
#         print(doc.doc_id, doc.doc_name)
#         doc.print_events()
#         build_giveme5w1h_training_dataset(doc, outputFolder="data/ECBplus_giveme5w1h")