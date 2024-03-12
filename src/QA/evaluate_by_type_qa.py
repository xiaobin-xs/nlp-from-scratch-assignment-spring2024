import csv
import argparse
from collections import defaultdict

from utils_eval import normalize_answer, f1_score, exact_match_score, rouge_score
from utils import load_txt_file_all_rows

parser = argparse.ArgumentParser(description="Evaluation RAG output for Anlp hw2")
parser.add_argument('--reference_ans_path', type=str, default="/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/reference_answers.txt")
parser.add_argument('--output_path', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/system_output_250_0_faiss_sentence-transformer_llama_20240311_022722.txt')
parser.add_argument('--qa_pair_path', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/qa-pair-lists.csv')
args = parser.parse_args()

# Load system outputs and reference answers
system_output = load_txt_file_all_rows(args.output_path)
reference_ans = load_txt_file_all_rows(args.reference_ans_path)
questions = load_txt_file_all_rows("/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt")  # Add the correct path

# Normalize text
system_output = [normalize_answer(ans) for ans in system_output]
reference_ans = [normalize_answer(ans) for ans in reference_ans]

# Load question categories
question_categories = {}
with open(args.qa_pair_path, mode='r') as infile:
    reader = csv.reader(infile)
    next(reader)  # Skip the header
    for row in reader:
        question_type, question, *_ = row
        question_categories[question] = question_type

# Initialize metrics by category
category_metrics = defaultdict(lambda: {'precision_recall_f1': [], 'em': [], 'rouge': []})


# Evaluate each question-answer pair
for i, question in enumerate(questions):
    category = question_categories[question]
    precision_recall_f1 = f1_score(system_output[i], [reference_ans[i]], return_precision_recall=True)
    em = exact_match_score(system_output[i], [reference_ans[i]])
    rouge = rouge_score(system_output[i], [reference_ans[i]])

    category_metrics[category]['precision_recall_f1'].append(precision_recall_f1)
    category_metrics[category]['em'].append(em)
    category_metrics[category]['rouge'].append(rouge)

# Save eval results to local file
file_path = args.output_path.replace("system_output","eval_result")

# Compute and print average metrics for each category
for category, metrics in category_metrics.items():
    avg_precision = sum([x[0] for x in metrics['precision_recall_f1']]) / len(metrics['precision_recall_f1'])
    avg_recall = sum([x[1] for x in metrics['precision_recall_f1']]) / len(metrics['precision_recall_f1'])
    avg_f1 = sum([x[2] for x in metrics['precision_recall_f1']]) / len(metrics['precision_recall_f1'])
    avg_em = sum(metrics['em']) / len(metrics['em'])
    ave_rouge_1 = sum([x[0] for x in metrics['rouge']]) / len(metrics['rouge'])
    ave_rouge_2 = sum([x[1] for x in metrics['rouge']]) / len(metrics['rouge'])
    ave_rouge_l = sum([x[2] for x in metrics['rouge']]) / len(metrics['rouge'])

    print(f'Category: {category}')
    print(f'Average Precision: {avg_precision}')
    print(f'Average Recall: {avg_recall}')
    print(f'Average F1: {avg_f1}')
    print(f'Average Exact Match: {avg_em}')
    print(f'Average Rouge-1: {ave_rouge_1}')
    print(f'Average Rouge-2: {ave_rouge_2}')
    print(f'Average Rouge-L: {ave_rouge_l}')
    print('---')
    
    # with open(file_path, 'a') as file:
    #     file.write(f'Category: {category}\n')
    #     file.write(f'Average Precision: {avg_precision}\n')
    #     file.write(f'Average Recall: {avg_recall}\n')
    #     file.write(f'Average F1: {avg_f1}\n')
    #     file.write(f'Average Exact Match: {avg_em}\n')
    #     file.write(f'Average Rouge-1: {ave_rouge_1}\n')
    #     file.write(f'Average Rouge-2: {ave_rouge_2}\n')
    #     file.write(f'Average Rouge-L: {ave_rouge_l}\n')
    #     file.write('---\n')
    
    


# Note: The answer to "When" question is significantly worse than other question
# Note: chunk size of 250 with k=10 can provide more related infomation, but lead to lower performance.