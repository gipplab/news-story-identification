'''
helper script to demonstrate the entire learn weights process pipeline.
CLEAR /queue_caches/ and result folder before running !!!!!!!!!!
Evaluate can`t distinguish between old and new results.
'''

import logging
import queue
import threading

import run_training, run_test
from run_training import Trainer
from run_test import Tester
from evaluate import process_files
from learn import Worker
# from run_test import load_best_weights, load_weights_csv

import argparse

def create_worker(q):
    for i in range(4):
        t = Worker(q)
        t.daemon = True
        t.start()


def get_queue_wth_lock_and_worker():
    # queue for multi threading support
    q = queue.Queue()
    # Wordnet is not threadsave...
    lock = threading.Lock()
    # Working threads
    create_worker(q)

    return q, lock


def load_trainer_for_question(q, questions, lock, trainer):
    for question in questions:
        if question == 'who' or question == 'what':
            q.put(trainer.action(lock))
        elif question == 'why':
            q.put(trainer.cause(lock))
        elif question == 'where':
            q.put(trainer.environment_where(lock))
        elif question == 'when':
            q.put(trainer.environment_when(lock))
        elif question == 'how':
            q.put(trainer.method(lock))
        elif question == 'cs':
            q.put(trainer.default_combined_scoring(lock))


def load_tester_for_question(q, questions, lock, tester):
    for question in questions:
        weights = tester.load_best_weights('./result/training_final_result_' + question + '_1.json')
        if question == 'who' or question == 'what':
            q.put(tester.action(lock, weights))
        elif question == 'why':
            q.put(tester.cause(lock, weights))
        elif question == 'where':
            q.put(tester.environment_where(lock, weights))
        elif question == 'when':
            q.put(tester.environment_when(lock, weights))
        elif question == 'how':
            q.put(tester.method(lock, weights))
        elif question == 'cs':
            q.put(tester.default_combined_scoring(lock, weights))

# Default dataset directory
OUTPUT_FOLDER = 'ECBplus_giveme5w1h'

def start_train(questions=[]):
    if len(questions) == 0:
        raise Exception(f'Questions must not be empty')

    log = logging.getLogger('GiveMe5W')
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    #
    # Training - to find the best weights
    #
    q, lock = get_queue_wth_lock_and_worker()

    trainer = Trainer()
    load_trainer_for_question(q, learn_questions, lock, trainer)

    # wait till all trainings are done
    q.join()
    print(f"======= DONE - TRAINING PHASE =======")
    print(f"======= START - TRAINING EVALUATION PHASE =======")
    #
    # Training - evaluate
    #

    # evaluate results - by cecking all subfolders for processd woking parts
    process_files('queue_caches/*_processed*/', praefix='training')

    print(f"======= DONE - TRAINING EVALUATION PHASE =======")
    print(f"======= START - TESTING PHASE =======")

    #
    # Test - with the best weights - found with training/evaluation
    #
    q, lock = get_queue_wth_lock_and_worker()

    tester = Tester()
    load_tester_for_question(q, learn_questions, lock, tester)
    q.join()
    print(f"======= DONE - TESTING PHASE =======")
    print(f"======= START - TESTING EVALUATION PHASE =======")
    #
    # Test - evaluate
    #
    process_files('queue_caches/*pre_calculated_processed*/', praefix='test')

    print('done')
    print(f"======= START - TESTING EVALUATION PHASE =======")

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Arguments parser')

    # Optional argument
    parser.add_argument(
        '--data-dir', type=str, nargs='?', const=f'{OUTPUT_FOLDER}', default=f'{OUTPUT_FOLDER}',
                        help='Path to dataset\'s location/directory')

    # Optional argument
    parser.add_argument(
        '--questions', type=str, nargs='*', default=['what'],
                        help="""Number of desired topics. Its recommended to run one by one to keep memory print low""")

    args = parser.parse_args()

    # who and what are using them same trainer. Declare just one !!!

    # its recommended to run one by one to keep memory print low
    # learn_questions = ['what']  # output is also  who
    # learn_questions = ['why']
    # learn_questions = ['where']
    # learn_questions = ['when']
    # learn_questions = ['how']
    learn_questions = [question for question in args.questions]
    print(f'Start training with questions {learn_questions}')

    start_train(learn_questions)

