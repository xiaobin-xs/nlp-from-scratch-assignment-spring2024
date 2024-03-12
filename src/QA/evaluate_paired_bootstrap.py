import argparse
import numpy as np

from utils_eval import normalize_answer, f1_score, exact_match_score, rouge_score
from utils import load_txt_file_all_rows

EVAL_TYPES = ['precision', 'recall', 'f1', 'exact_match', 'rouge_1', 'rouge_2', 'rouge_l']


def eval_measure(gold, sys, eval_type='precision'):
    ''' Evaluation measure
    
    This takes in gold labels and system outputs and evaluates their performance. 

    :param gold: the correct labels
    :param sys: the system outputs
    :param eval_type: The type of evaluation to do (precision, recall, f1, exact_match, rouge_1, rouge_2, rouge_l)
    '''
    if eval_type == 'precision':
        precision_recall_f1_list = [f1_score(sys[i], [gold[i]], return_precision_recall=True) for i in range(len(sys))]
        avg_precision = sum([x[0] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
        return avg_precision
    elif eval_type == 'recall':
        precision_recall_f1_list = [f1_score(sys[i], [gold[i]], return_precision_recall=True) for i in range(len(sys))]
        avg_recall = sum([x[1] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
        return avg_recall
    elif eval_type == 'f1':
        precision_recall_f1_list = [f1_score(sys[i], [gold[i]], return_precision_recall=True) for i in range(len(sys))]
        avg_f1 = sum([x[2] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
        return avg_f1
    elif eval_type == 'exact_match':
        em_list = [exact_match_score(sys[i], [gold[i]]) for i in range(len(sys))]
        avg_em = sum(em_list) / len(em_list)
        return avg_em
    elif eval_type == 'rouge_1':
        rouge_list = [rouge_score(sys[i], [gold[i]]) for i in range(len(sys))]
        ave_rouge_1 = sum([x[0] for x in rouge_list]) / len(rouge_list)
        return ave_rouge_1
    elif eval_type == 'rouge_2':
        rouge_list = [rouge_score(sys[i], [gold[i]]) for i in range(len(sys))]
        ave_rouge_2 = sum([x[1] for x in rouge_list]) / len(rouge_list)
        return ave_rouge_2
    elif eval_type == 'rouge_l':
        rouge_list = [rouge_score(sys[i], [gold[i]]) for i in range(len(sys))]
        ave_rouge_l = sum([x[2] for x in rouge_list]) / len(rouge_list)
        return ave_rouge_l
    else:
        raise NotImplementedError('Unknown eval type in eval_measure: %s' % eval_type)
    

def eval_with_paired_bootstrap(gold, sys1, sys2,
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='precision'):
    ''' Evaluate with paired boostrap

    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the accuracy of the two systems.
    
    :param gold: The correct labels
    :param sys1: The output of system 1
    :param sys2: The output of system 2
    :param num_samples: The number of bootstrap samples to take
    :param sample_ratio: The ratio of samples to take every time
    :param eval_type: The type of evaluation to do (precision, recall, f1, exact_match, rouge_1, rouge_2, rouge_l)
    '''
    assert(len(gold) == len(sys1))
    assert(len(gold) == len(sys2))
    
    # # Preprocess the data appropriately for they type of eval
    # gold = [eval_preproc(x, eval_type) for x in gold]
    # sys1 = [eval_preproc(x, eval_type) for x in sys1]
    # sys2 = [eval_preproc(x, eval_type) for x in sys2]

    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0]
    n = len(gold)
    ids = list(range(n))

    for _ in range(num_samples):
        # Subsample the gold and system outputs
        reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio),replace=True)
        reduced_gold = [gold[i] for i in reduced_ids]
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]
        # Calculate accuracy on the reduced sample and save stats
        sys1_score = eval_measure(reduced_gold, reduced_sys1, eval_type=eval_type)
        sys2_score = eval_measure(reduced_gold, reduced_sys2, eval_type=eval_type)
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    # Print win stats
    wins = [x/float(num_samples) for x in wins]
    print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
    elif wins[1] > wins[0]:
        print('(sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
                    (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
                    (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
    


exp_name1 = '250_0_faiss_vecstore_10_sentence-transformer_gpt4all_0-shot_20240312_152816'
exp_name2 = '500_0_faiss_vecstore_4_sentence-transformer_gpt4all_fewshot0_20240311_234553'
reference_ans_path = '../../data/test/reference_answers.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--reference_ans_path', help='File of the correct answers', default=reference_ans_path)
parser.add_argument('--exp_name1', help='Experiment name for system 1', default=exp_name1)
parser.add_argument('--exp_name2', help='Experiment name for system 2', default=exp_name2)
parser.add_argument('--eval_type', help='The evaluation type (precision, recall, f1, exact_match, rouge_1, rouge_2, rouge_l)', type=str, default='f1', choices=EVAL_TYPES)
parser.add_argument('--num_samples', help='Number of samples to use', type=int, default=10000)
args = parser.parse_args()

system_output_path1 = '../../data/test/' + f'system_output_{args.exp_name1}.txt'
system_output_path2 = '../../data/test/' + f'system_output_{args.exp_name2}.txt'
reference_ans_path = args.reference_ans_path


system_output1 = load_txt_file_all_rows(system_output_path1)
system_output2 = load_txt_file_all_rows(system_output_path2)
reference_ans = load_txt_file_all_rows(reference_ans_path)

system_output1 = [normalize_answer(ans) for ans in system_output1]
system_output2 = [normalize_answer(ans) for ans in system_output2]
reference_ans = [normalize_answer(ans) for ans in reference_ans]

print('System 1:', args.exp_name1)
print('System 2:', args.exp_name2)
print('Evaluation type:', args.eval_type)
eval_with_paired_bootstrap(reference_ans, system_output1, system_output2, 
                           eval_type=args.eval_type, num_samples=10000, sample_ratio=0.5)