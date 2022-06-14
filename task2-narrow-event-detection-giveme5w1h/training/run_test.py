"""
Helper to run learn with different configurations.
Be sure to remove the training set from input before running
"""
import csv
import json
import logging
import queue
import threading

from extractor.root import path
from extractor.extractors import environment_extractor, action_extractor, cause_extractor, method_extractor
from learn import Learn, Worker
from work_queue import WorkQueue

## OLD CODE ##
# dataFolder = 'gold_standard_small'
# dataFolder = 'gold_standard'
# inputPath = path(f'../examples/datasets/{dataFolder}/data')
# preprocessedPath = path(f'../examples/datasets/{dataFolder}/cache')
## OLD CODE ##

class Tester():

    def __init__(self, dataFolder='ECBplus_giveme5w1h'):
        self.dataFolder = dataFolder
        self.inputPath = path(f'../examples/datasets/{dataFolder}/data')
        self.preprocessedPath = path(f'../examples/datasets/{dataFolder}/cache')

    def load_best_weights(self, path):
        with open(path, encoding='utf-8') as data_file:
            data = json.load(data_file)
        return data['best_dist']['weights']


    def load_weights_csv(self, path, top_percent: float=0.05):
        csv_content = []

        header = True
        print(f'{path}.csv')
        with open(path + '.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            # read header

            # read file
            for row in reader:
                if header:
                    header = False
                    score_index = row.index("score")
                else:
                    csv_content.append(row)

        # should already be sorted, but to be sure....
        # csv_content.sort(key=lambda x: x[score_index])
        take_top_n = int(len(csv_content) * top_percent)
        last_weight = score_index

        sub_list = csv_content[0: take_top_n]
        final_list = [o[0:last_weight] for o in sub_list]

        # string to float
        for i,weights in enumerate(final_list):
            for ix, weight in enumerate(weights):
                final_list[i][ix] = float(final_list[i][ix])
        return final_list


    def method(self, lock, pre_calculated_weights):
        a_queue = WorkQueue(id='test_method', generator='pre_calculated', pre_calculated_weights=pre_calculated_weights)
        a_queue.setup_scoring_parameters()
        a_queue.setup_extracting_parameters()
        a_queue.load()

        extractors = {
            'method': method_extractor.MethodExtractor()
        }
        learn = Learn(lock=lock,
                    extractors=extractors,
                    preprocessed_path=self.preprocessedPath,
                    input_path=self.inputPath,
                    sampling='test',
                    combined_scorer=None, queue=a_queue)
        return learn


    def cause(self, lock, pre_calculated_weights):
        a_queue = WorkQueue(id='test_cause', generator='pre_calculated', pre_calculated_weights=pre_calculated_weights)
        a_queue.load()

        extractors = {
            'cause': cause_extractor.CauseExtractor()
        }

        learn = Learn(lock=lock,
                    extractors=extractors,
                    preprocessed_path=self.preprocessedPath,
                    input_path=self.inputPath,
                    sampling='test',
                    combined_scorer=None, queue=a_queue)
        return learn


    def environment_where(self, lock, pre_calculated_weights):
        a_queue = WorkQueue(id='test_environment_where', generator='pre_calculated',
                            pre_calculated_weights=pre_calculated_weights)

        a_queue.load()

        extractors = {
            'environment_where': environment_extractor.EnvironmentExtractor(skip_when=True)
        }

        learn = Learn(lock=lock,
                    extractors=extractors,
                    preprocessed_path=self.preprocessedPath,
                    input_path=self.inputPath,
                    sampling='test',
                    combined_scorer=None, queue=a_queue)
        return learn


    def environment_when(self, lock, pre_calculated_weights):
        a_queue = WorkQueue(id='test_environment_when', generator='pre_calculated',
                            pre_calculated_weights=pre_calculated_weights)

        a_queue.load()

        extractors = {
            'environment_when': environment_extractor.EnvironmentExtractor(skip_where=True)
        }

        learn = Learn(lock=lock,
                    extractors=extractors,
                    preprocessed_path=self.preprocessedPath,
                    input_path=self.inputPath,
                    sampling='test',
                    combined_scorer=None, queue=a_queue)
        return learn


    def action(self, lock, pre_calculated_weights):
        a_queue = WorkQueue(id='test_action', generator='pre_calculated', pre_calculated_weights=pre_calculated_weights)
        a_queue.setup_scoring_parameters()
        a_queue.setup_extracting_parameters()
        a_queue.load()

        extractors = {
            'action': action_extractor.ActionExtractor()
        }
        learn = Learn(lock=lock,
                    extractors=extractors,
                    preprocessed_path=self.preprocessedPath,
                    input_path=self.inputPath,
                    sampling='test',
                    combined_scorer=None, queue=a_queue)
        return learn


# if __name__ == '__main__':

#     log = logging.getLogger('GiveMe5W')
#     log.setLevel(logging.INFO)
#     sh = logging.StreamHandler()
#     sh.setLevel(logging.INFO)
#     log.addHandler(sh)

#     # thread safe queue
#     q = queue.Queue()
#     lock = threading.Lock()  # Wordnet is not threadsave

#     #print()
#     # WHO, WHAT
#     weights = load_best_weights('./result/training_final_result_what_1.json')
#     print(f"==== DEBUG  Best weights WHO & WHAT {weights}")
#     input("Press enter to continue...")
#     q.put(action(lock, weights))

#     # WHERE
#     weights = load_best_weights('./result/training_final_result_where_1.json')
#     print(f"==== DEBUG  Best weights WHO & WHAT {weights}")
#     input("Press enter to continue...")
#     q.put(environment_where(lock, weights))

#     # WHEN
#     """ FROM """
#     #weights = load_weights_csv('./result/wmd/wmd_when_fix_entailment/training_final_result_when_1_avg')
#     # weights = load_weights_csv('./result/training_final_result_when_1_avg')
#     weights = load_best_weights('./result/training_final_result_when_1.json')
#     print(f"==== DEBUG  Best weights WHO & WHAT {weights}")
#     input("Press enter to continue...")
#     q.put(environment_when(lock, weights))

#     # WHY
#     weights = load_best_weights('./result/training_final_result_why_1.json')
#     print(f"==== DEBUG  Best weights WHO & WHAT {weights}")
#     input("Press enter to continue...")
#     q.put(cause(lock, weights))

#     # HOW
#     weights = load_best_weights('./result/training_final_result_how_1.json')
#     print(f"==== DEBUG  Best weights WHO & WHAT {weights}")
#     input("Press enter to continue...")
#     q.put(method(lock, weights))

#     for i in range(4):
#         t = Worker(q)
#         t.daemon = True
#         t.start()

#     # wait till all extractors are done
#     q.join()