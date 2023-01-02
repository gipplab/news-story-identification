import csv
import logging
import os
import random
from datetime import datetime

from sentence_transformers import (LoggingHandler, SentenceTransformer,
                                   evaluation, losses, util)
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

# Switches
paraphraseClassificationTraining = True
duplicateMiningTraining = False
informationRetrievalTraining = False
openShell = True

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


#As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model = SentenceTransformer('stsb-distilbert-base')
num_epochs = 2
train_batch_size = 64

#As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

#Negative pairs should have a distance of at least 0.5
margin = 0.5

dataset_path = './data/quora-IR-dataset'
model_save_path = './model/training_OnlineConstrativeLoss-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)

######### Read train data  ##########
# Read train data
train_samples = []
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sample = InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate']))
        train_samples.append(sample)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
if paraphraseClassificationTraining:
    dev_sentences1 = []
    dev_sentences2 = []
    dev_labels = []
    with open(os.path.join(dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            dev_sentences1.append(row['question1'])
            dev_sentences2.append(row['question2'])
            dev_labels.append(int(row['is_duplicate']))


    binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)
    evaluators.append(binary_acc_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)
    
# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
    evaluator=seq_evaluator,
    epochs=num_epochs,
    warmup_steps=1000,
    output_path=model_save_path
)

if openShell:
    import code
    code.interact(local=locals())