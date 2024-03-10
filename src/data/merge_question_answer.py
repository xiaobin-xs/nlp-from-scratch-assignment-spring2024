def merge_question_answer_files(question_file_path, answer_file_path, output_file_path):
    with open(question_file_path, 'r') as q_file, open(answer_file_path, 'r') as a_file, open(output_file_path, 'w') as output_file:
        questions = q_file.readlines()
        answers = a_file.readlines()
        if len(questions) != len(answers):
            print("Error: The number of questions and answers does not match.")
            return

        for question, answer in zip(questions, answers):
            question = question.strip()
            answer = answer.strip()
            output_file.write(f"Question: {question}\nAnswer: {answer}.\n\n")

question_file_path = '/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt'
answer_file_path = '/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/reference_answers.txt'
output_file_path = '/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/question_answers.txt'

merge_question_answer_files(question_file_path, answer_file_path, output_file_path)
