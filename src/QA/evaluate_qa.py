import argparse

from utils_eval import normalize_answer, f1_score, exact_match_score, rouge_score
from utils import load_txt_file_all_rows


parser= argparse.ArgumentParser(description="Evaluation RAG output for Anlp hw2")
parser.add_argument('--reference_ans_path',type=str, default="/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/reference_answers.txt")
parser.add_argument('--output_path', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/system_output_250_0_faiss_sentence-transformer_llama_small_chunk_higher_K_20240312_053206.txt')
args = parser.parse_args()

reference_ans_path = args.reference_ans_path
output_path = args.output_path

## load system output and reference answers
system_output = load_txt_file_all_rows(output_path)
reference_ans = load_txt_file_all_rows(reference_ans_path)
## normalize text
system_output = [normalize_answer(ans) for ans in system_output]
reference_ans = [normalize_answer(ans) for ans in reference_ans]

precision_recall_f1_list = [f1_score(system_output[i], [reference_ans[i]], return_precision_recall=True) for i in range(len(system_output))]
em_list = [exact_match_score(system_output[i], [reference_ans[i]]) for i in range(len(system_output))]
rouge_list = [rouge_score(system_output[i], [reference_ans[i]]) for i in range(len(system_output))]


avg_precision = sum([x[0] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
avg_recall = sum([x[1] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
avg_f1 = sum([x[2] for x in precision_recall_f1_list]) / len(precision_recall_f1_list)
avg_em = sum(em_list) / len(em_list)
ave_rouge_1 = sum([x[0] for x in rouge_list]) / len(rouge_list)
ave_rouge_2 = sum([x[1] for x in rouge_list]) / len(rouge_list)
ave_rouge_l = sum([x[2] for x in rouge_list]) / len(rouge_list)

print(f'Average Precision: {avg_precision}')
print(f'Average Recall: {avg_recall}')
print(f'Average F1: {avg_f1}')
print(f'Average Exact Match: {avg_em}')
print(f'Average Rouge-1: {ave_rouge_1}')
print(f'Average Rouge-2: {ave_rouge_2}')
print(f'Average Rouge-L: {ave_rouge_l}')

# python3 evaluate_by_type_qa.py --output_path /home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/system_output_500_0_chroma_sentence-transformer_llama_20240311_000912.txt