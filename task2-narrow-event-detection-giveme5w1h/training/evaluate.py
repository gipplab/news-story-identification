"""
checks all result files  subdirectories of queue_caches an writes results to /result/
"""
import csv
import glob
import json
import os
import pickle
import statistics
from itertools import groupby

from csv_to_parallel_coordinates_plotter import generate_plot
from extractor.tools import mapper

def weights_to_string(weights):
    """
    converts an array of ints to a string.
    :param weights:
    :return:
    """
    scaled_weights_string = [str(x) for x in weights]
    return '_'.join(scaled_weights_string)


def process_files(path, praefix):
    """
    reads all processed q items an merges them into one dict
    :param path:
    :return:
    """
    score_results = {}

    # walk over app items directories
    for directory_path in glob.glob(path):
        if praefix is None or (praefix and directory_path.find(praefix) != -1):
            # walk over all parts
            entire_qu = []
            for file_path in glob.glob(directory_path + '/*'):
                print(f'==== Gonna read file {file_path}')
                if os.path.getsize(file_path) > 0:
                    with open(file_path, 'rb') as ff:
                        processed_item = pickle.load(ff)
                        entire_qu.extend(processed_item)

            # merge qu to one dict, merge per question and weight
            for result in entire_qu:
                # walk over each result object,
                # each extractor can answer more than one question
                for question in result['result']:
                    question_scores = score_results.setdefault(question, {})
                    weights = result['result'][question][1]

                    # create a identifier for these weights
                    weights_string = weights_to_string(weights)

                    # each item is identified by their extracting parameters, weight
                    # and their answer (stored over the parent node)
                    comb_for_this_parameter_id = question_scores.setdefault(result['extracting_parameters_id'], {
                        'extracting_parameters': result['extracting_parameters'], 'weights': {}})

                    comb = comb_for_this_parameter_id['weights'].setdefault(weights_string,
                                                                            {'weights': weights, 'scores_doc': []})

                    # save this score to all results
       
                    comb['scores_doc'].append(result['result'][question][2])
    """ FROM # Changed write_full=False to True"""
    evaluate(score_results, write_full=True, praefix=praefix)


def remove_errors(list):
    """
    returns a list where all -1 are replace with the biggest value
    :param list:
    :return:
    """
    # remove no annotation error, by replacing with worst distance
    a_max = max(list)
    error_count = 0
    tmp = []
    for score in list:
        if score <= 0:
            tmp.append(a_max)
            error_count = error_count + 1
        else:
            tmp.append(score)

    return tmp, error_count, a_max


def normalize(list):
    """
    this is assuming that there is a min with 0
    :param list:
    :return:
    """
    # find max
    a_max = max(list)
    # set errors to max
    list_error_free = [x if x >= 0 else a_max for x in list]
    # normalize
    result = []
    for entry in list_error_free:
        result.append(entry / a_max)

    return result


def merge_top(a_list, accessor):
    """
    multiple weights can produce the same top-score, this function merges all top weights.
    :param a_list: 
    :param accessor: 
    :return: 
    """
    a_list.sort(key=lambda x: x['score'], reverse=False)
    result = a_list[0]
    weights = []
    result['weights'] = weights

    for entry in a_list:
        if entry[accessor] == result[accessor]:
            a_weight = entry.get('weight')
            if a_weight:
                weights.append(a_weight)
        else:
            break
    return result


def to_ranges(iterable):
    """
    finds range of ints in a list of ints, this returns a generator!!
    :param iterable:
    :return:
    """
    iterable = sorted(set(iterable))
    for key, group in groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def to_ranges_wrapper(iterable, decimals: int = 10):
    """
    converts a list of float to ints and finds range in this list
    :param iterable:
    :return:
    """

    for ita, i in enumerate(iterable):
        iterable[ita] = int(iterable[ita] * decimals)

    result = list(to_ranges(iterable))

    return result


def stats_helper(list):
    """
    https://docs.python.org/3/library/statistics.html#statistics.pvariance
    :param list:
    :return:
    """

    mean = statistics.mean(list)
    mode = None

    try:
        mode = statistics.mode(list)
    except statistics.StatisticsError:
        # no unique mode
        pass

    return {
        'mean': mean,
        'variance': statistics.pvariance(list, mu=mean),
        'standard_deviation': statistics.pstdev(list, mu=mean),
        'median': statistics.median(list),
        'median_low': statistics.median_low(list),
        'median_high': statistics.median_high(list),
        'median_grouped': statistics.median_grouped(list),
        'mode': mode
    }


def generate_csv(praefix, score_per_average, accessor='avg'):
    # write combinations per extractor to CSV
    for question in score_per_average:
        csv_results_avg = []
        csv_results_all = []
        weights = list(score_per_average[question].values())
        extractor_name = mapper.question_to_extractor(question.split('_')[0])

        # take first weight to form a header line
        headerline = []
        for i, weight in enumerate(weights[0]['weight']):
            headerline.append(mapper.weight_to_string(extractor_name, i, question=question))
        headerline.append('score')
        csv_results_avg.append(headerline)
        csv_results_all.append(headerline)

        for weight in weights:
            # average score over all document off these weights
            csv_results_avg.append(list(weight['weight']) + [weight[accessor]])

        filename = praefix + '_final_result_' + question + '_' + accessor
        print(f"===== FILE NAME {filename} {question} {accessor}")
        with open('D:/Projects/MasterThesis/task2-narrow-event-detection-giveme5w1h/training/result/' + filename + '.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for line in csv_results_avg:
                writer.writerow(line)
        # generate plot files
        generate_plot(filename, auto_open=False)


def golden_weights_to_ranges(a_list):
    """
    converts golden weights to ranges per weight to make importance more visible
    [0.1, 0.1] [0.2, 0.1] [0.3, 0.9]

    this function works only well with 0.1 step weights, result is not  converted back to float
    ---->
    {
        0: [1 - 3]
        1: [1 - 1] [0.9 - 0.9]
    }
    "EXPERIMENTAL - NOT VERY WELL TESTED"

    :param a_list: 
    :return: 
    """
    golden_weights = a_list.get('best_dist')['weights']
    if golden_weights and len(golden_weights) > 0:
        # slots for each weight
        weights = [[] for _ in range(len(golden_weights[0]))]

        # copy weights based on their location into new format
        for combination in golden_weights:
            for i, weight in enumerate(combination):
                weights[i].append(weight)

        # find the ranges per weight location
        results = []
        for weight in weights:
            uniqu_weights = list(set(weight))
            uniqu_weights.sort()

            ranges = to_ranges_wrapper(uniqu_weights)
            results.append({
                'stats': stats_helper(weight),
                'data': ranges
            })

        a_list['golden_groups'] = results


def index_of_best(list):
    """
    low distance is better
    :param list:
    :return:
    """
    a_list, error_count, error = remove_errors(list)
    return list.index(min(a_list))


def evaluate(score_results, write_full: bool = False, praefix=''):
    # write raw results
    if write_full:
        with open('D:/Projects/MasterThesis/task2-narrow-event-detection-giveme5w1h/training/result/' + praefix + '_evaluation_full' + '.json', 'w+') as data_file:
            data_file.write(json.dumps(score_results, sort_keys=False, indent=4))
            data_file.close()

    # has a low dist on average per weight (documents are merged)
    score_per_average_extrem = {}
    score_per_average = {}

    for question in score_results:
        score_per_average_extrem
        for extracting_parameters_id in score_results[question]:
            question_extract_id = question + '_' + str(extracting_parameters_id)
            extr = score_per_average_extrem.setdefault(question_extract_id, {
                'min': float('inf'),
                'max': float('-inf')
            })
            for combination_string in score_results[question][extracting_parameters_id]['weights']:
                combo = score_results[question][extracting_parameters_id]['weights'][combination_string]

                raw_scores = combo['scores_doc']
                scores, error_count, error = remove_errors(raw_scores)

                # error is (error_count * error) - subtract it to keep score representative (e.g. km)
                scores_sum_wrong = sum(scores)
                scores_sum = scores_sum_wrong - (error_count * error)

                avg = (scores_sum / len(scores))

                score_per_average.setdefault(question_extract_id, {})[combination_string] = {
                    'score': scores_sum,
                    'error_count': error_count,
                    'avg': avg,
                    'weight': combo['weights']
                }

                extr['min'] = min(extr['min'], scores_sum)
                extr['max'] = max(extr['max'], scores_sum)

    for min_max in score_per_average_extrem:
        min_max = score_per_average_extrem[min_max]
        min_max['max_minus_min'] = min_max['max'] - min_max['min']



    # normalize the avg weights over all weights, per question
    for question in score_per_average:
        items = score_per_average[question]
        extrem_item = score_per_average_extrem[question]
        for item_id in items:
            item = items[item_id]
            if extrem_item['max_minus_min'] != 0:
                item['norm_score'] = (item['score'] - extrem_item['min']) / extrem_item['max_minus_min']
            else:
                item['norm_score'] = (item['score'] - extrem_item['min'])

    # sort all scores
    #for question in score_per_average:
     #   items = score_per_average[question]
        #items.sort(key=lambda x: x['score'])

    # finally, get the best weighting and save it to a file
    final_result = {}
    for question in score_per_average:
        score_per_average_list = list(score_per_average[question].values())

        final_result[question] = {
            'best_dist': merge_top(score_per_average_list, 'norm_score')
        }

        golden_weights_to_ranges(final_result[question])

    # write it as json
    for question in final_result:
        with open('D:/Projects/MasterThesis/task2-narrow-event-detection-giveme5w1h/training/result/' + praefix + '_final_result_' + question + '.json', 'w+') as data_file:
            data_file.write(json.dumps(final_result[question], sort_keys=False, indent=4))
            data_file.close()
    generate_csv(praefix, score_per_average, 'avg')
    generate_csv(praefix, score_per_average, 'norm_score')


if __name__ == '__main__':
    # evaluate training
    process_files('queue_caches/*_processed*/', praefix='training')
    process_files('queue_caches/*pre_calculated_processed*/', praefix='test')
    # process_files('queue_caches/*where_pre_calculated_processed*/', praefix='test')