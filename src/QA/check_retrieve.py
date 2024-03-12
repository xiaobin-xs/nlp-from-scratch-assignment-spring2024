import csv
import re
from utils_eval import normalize_answer, f1_score, exact_match_score, rouge_score

def load_correct_answers(csv_file_path):
    """
    Load correct answers from a CSV file.
    """
    correct_answers = {}
    with open(csv_file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            correct_answers[normalize_answer(row['question']).lower()] = normalize_answer(row['answer']).lower()
    return correct_answers

def load_retrieval_results(retrieval_results_file_path):
    """
    Load retrieval results from a text file.
    """
    with open(retrieval_results_file_path, 'r') as file:
        retrieval_results_text = file.read()
    return retrieval_results_text

def check_answer_presence(retrieval_result, correct_answers,answered_questions_file_path , unanswered_questions_file_path):
    """
    Check if the correct answer exists within the retrieved document content.
    """
    pattern = r"Question: (.*?)\n\tNumber of retrieved documents: \d+(.*?)\n-{10,}"
    results = re.findall(pattern, retrieval_result, re.DOTALL)
    
    correct_retrieval_count = 0
    unanswered_questions = []
    answered_questions = []
    
    for result in results:
        question, documents = result
        normalized_question = normalize_answer(question).lower()
        correct_answer = correct_answers.get(normalized_question)
        
        if correct_answer and any(correct_answer in doc.lower() for doc in documents.split('Document Name:')):
            correct_retrieval_count += 1
            answered_questions.append(question)
        else:
            unanswered_questions.append(question)
    
    with open(answered_questions_file_path, 'w') as file:
        for question in answered_questions:
            file.write(question + '\n')

    with open(unanswered_questions_file_path, 'w') as file:
        for question in unanswered_questions:
            file.write(question + '\n')
    
    return correct_retrieval_count, len(results)

def main(retrieval_results_file_path, csv_file_path, answered_questions_file_path, unanswered_questions_file_path):
    correct_answers = load_correct_answers(csv_file_path)
    retrieval_results_text = load_retrieval_results(retrieval_results_file_path)
    correct_count, total_questions = check_answer_presence(retrieval_results_text, correct_answers, answered_questions_file_path, unanswered_questions_file_path)
    if total_questions > 0:
        proportion_correct = correct_count / total_questions
        print(f"Proportion of correctly retrieved results: {proportion_correct:.2f}")
    else:
        print("No questions found in the retrieval results.")

# Specify the paths to your files
retrieval_results_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/log/retrieval_result_1000_0_faiss_sentence-transformer_llama_20240312_035304.txt"
csv_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/qa-pair-lists.csv"
answered_questions_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/answered_questions_llama.txt"
unanswered_questions_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/unanswered_questions_llama.txt"
main(retrieval_results_file_path, csv_file_path, answered_questions_file_path, unanswered_questions_file_path)