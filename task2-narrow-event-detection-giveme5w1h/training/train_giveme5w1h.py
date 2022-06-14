"""
Helper to run learn with different configurations.
Be sure to remove the training set from input before running
"""
import logging
import queue
import threading

from extractor.combined_scoring import distance_of_candidate
from extractor.root import path
from extractor.extractors import environment_extractor, action_extractor, cause_extractor, method_extractor
from learn import Learn, Worker
from work_queue import WorkQueue

dataFolder = 'gold_standard_small'
inputPath = path(f'../examples/datasets/{dataFolder}/data')
preprocessedPath = path(f'../examples/datasets/{dataFolder}/cache')
print(inputPath)
print(preprocessedPath)

def method(lock):
    a_queue = WorkQueue(id='training', generator='method')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'method': method_extractor.MethodExtractor()
    }
    learn = Learn(lock=lock,
                  extractors=extractors,
                  preprocessed_path=preprocessedPath,
                  input_path=inputPath,
                  sampling='training',
                  combined_scorer=None, queue=a_queue)
    return learn


def cause(lock):
    a_queue = WorkQueue(id='training', generator='cause')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'cause': cause_extractor.CauseExtractor()
    }

    learn = Learn(lock=lock,
                  extractors=extractors,
                  preprocessed_path=preprocessedPath,
                  input_path=inputPath,
                  sampling='training',
                  combined_scorer=None, queue=a_queue)
    return learn


def environment_where(lock):
    a_queue = WorkQueue(id='training', generator='environment_where')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'environment_where': environment_extractor.EnvironmentExtractor(skip_when=True)
    }

    learn = Learn(lock=lock, extractors=extractors, preprocessed_path=preprocessedPath, input_path=inputPath,
                  combined_scorer=None, queue=a_queue)
    return learn


def environment_when(lock):
    a_queue = WorkQueue(id='training', generator='environment_when')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'environment_when': environment_extractor.EnvironmentExtractor(skip_where=True)
    }

    learn = Learn(lock=lock, extractors=extractors, preprocessed_path=preprocessedPath, input_path=inputPath,
                  combined_scorer=None, queue=a_queue)
    return learn


def environment(lock):
    a_queue = WorkQueue(id='training', generator='environment')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'environment': environment_extractor.EnvironmentExtractor()
    }

    learn = Learn(lock=lock, extractors=extractors, preprocessed_path=preprocessedPath, input_path=inputPath,
                  combined_scorer=None, queue=a_queue)
    return learn


def action(lock):
    a_queue = WorkQueue(id='training', generator='action')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'action': action_extractor.ActionExtractor()
        # 'environment': environment_extractor.EnvironmentExtractor(),
        # 'cause': cause_extractor.CauseExtractor(),
        # 'method': method_extractor.MethodExtractor()
    }
    learn = Learn(lock=lock,
                  extractors=extractors,
                  preprocessed_path=preprocessedPath,
                  input_path=inputPath,
                  sampling='training',
                  combined_scorer=None, queue=a_queue)
    return learn


def default_combined_scoring(lock):
    a_queue = WorkQueue(id='training_cs', generator='combined_scoring')
    a_queue.setup_scoring_parameters()
    a_queue.setup_extracting_parameters()
    a_queue.load()

    extractors = {
        'method': method_extractor.MethodExtractor(),
        'action': action_extractor.ActionExtractor()  # provider for what an who
    }
    # set optimal weights learned beforehand
    # extractors['action'].weights = [0.7, 0.3, 0.9]

    combined_scorer = distance_of_candidate.DistanceOfCandidate(['what'], 'how')

    learn = Learn(lock=lock,
                  extractors=extractors,
                  preprocessed_path=preprocessedPath,
                  input_path=inputPath,
                  sampling='training',
                  combined_scorer=combined_scorer,
                  queue=a_queue,
                  ignore_extractor=['action'])
    return learn


if __name__ == '__main__':

    log = logging.getLogger('GiveMe5W')
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    # basic learner
    # log.setLevel(logging.ERROR)

    # thread safe queue
    q = queue.Queue()
    lock = threading.Lock()  # Wordnet is not threadsave

    # q.put(action(lock))
    # q.put(environment(lock))

    # q.put(environment_when(lock))
    #q.put(environment_where(lock))

    # q.put(cause(lock))
    # q.put(method(lock))
    q.put(default_combined_scoring(lock))

    for i in range(4):
        t = Worker(q)
        t.daemon = True
        t.start()

    # wait till all extractors are done
    q.join()