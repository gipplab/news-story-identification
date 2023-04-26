import code
import time
import warnings
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import consts
def read_data(folder, files):
    li = []
    for filename in files:
        print(f'=== Reading {filename}')
        df = pd.read_csv(join(folder, filename), index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    return df

def train(df):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings('ignore', message="Precision")

    train, test = train_test_split(df.reset_index(drop=True), test_size=.20, random_state=5)

    #The Doc2Vec model takes 'tagged_documents'
    #tag the training data
    tagged_tr = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train.text)]

    #tag testing data
    tagged_test = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(test.text)]

    #Instantiate the model

    model = Doc2Vec(vector_size=100, # 100 should be fine based on the standards
                    window=5, #change to 8
                    alpha=.025, #initial learning rate
                    min_alpha=0.00025, #learning rate drops linearly to this
                    min_count=2, #ignores all words with total frequency lower than this.
                    dm =1, #algorith 1=distributed memory
                    workers=16)#cores to use

    #build the vocab on the training data
    model.build_vocab(tagged_tr)

    #max training epochs
    max_epochs = 5

    #train n epochs and save the model
    t1 = time.time()
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch+1))
        model.train(tagged_tr,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    
    

    print("done!")
    t2 = time.time()    
    model.save(consts.Doc2Vec_Polarity_classifier_filepath)
    #print("Model Saved")
    print("Time: {}".format(t2-t1))

    #Now that we have the embedding trained, we can use the infer vector method to convert the test sentences into vectors
    #that can be used to model 

    # Extract vectors from doc2vec model
    X_train = np.array([model.docvecs[str(i)] for i in range(len(tagged_tr))])
    y_train = train.label

    # Extract test values
    X_test = np.array([model.infer_vector(tagged_test[i][0]) for i in range(len(tagged_test))])
    y_test = test.label

    lrc = LogisticRegression(C=5, multi_class='multinomial', solver='saga',max_iter=1000)
    lrc.fit(X_train,y_train)
    y_pred = lrc.predict(X_test)
    heatconmat(y_test,y_pred)


def heatconmat(y_true,y_pred):
    sns.set_context('talk')
    plt.figure(figsize=(9,6))
    sns.heatmap(confusion_matrix(y_true,y_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='gist_earth_r',
                yticklabels=sorted(y_true.unique()))
    plt.show()
    print(classification_report(y_true,y_pred))

if __name__ == "__main__":
    POLUSA = {
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
            'FOLDER': './data/polusa_balanced_432k',
            'FILES': ['data.csv']
        },
        
        'Full': {    
            'FOLDER': './data/polusa/polusa_balanced',
            'FILES':  ['2017_1.csv', '2017_2.csv', '2018_1.csv', '2018_2.csv', '2019_1.csv', '2019_2.csv']
        }
    }

    POLUSA_VERSION = '90k'   # <-- choose version here
    FOLDER = POLUSA[POLUSA_VERSION]['FOLDER']
    FILES = POLUSA[POLUSA_VERSION]['FILES']

    df = read_data(FOLDER, FILES)
    df = df.drop(df[df.label == 'UNDEFINED'].index)
    df = df.drop(['id', 'date_publish', 'outlet', 'headline', 'lead', 'authors', 'domain', 'url'], axis=1)

    trainning = True
    openShell = True
    
    if trainning:
        train(df)
    if openShell:
        code.interact(local=locals())
    