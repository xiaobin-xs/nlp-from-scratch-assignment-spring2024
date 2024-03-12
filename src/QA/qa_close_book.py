from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain import hub
import random
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os, sys
import argparse
from tqdm import tqdm
from utils import load_random_question, load_txt_file_all_rows, generate_answer, clean_output_answer
from datetime import datetime


chunk_size = 500 # e.g. 300, 500, 1000
chunk_overlap = 0
vecstore = "faiss" # 'faiss' or 'chroma'
retrieve_k_docs = 5
embed_model = 'sentence-transformer' # 'gpt4all', 'llama', or 'sentence-transformer'
llm = "llama" # 'gpt4all' or 'llama'
if llm == 'llama':
    model_path = '../../model/llama-2-13b-chat.Q4_K_M.gguf'
elif llm == 'gpt4all':
    model_path = '../../model/gpt4all-13b-snoozy-q4_0.gguf'
else:
    raise ValueError(f'llm model {llm} not defined')

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")
parser= argparse.ArgumentParser(description="RAG QA system for Anlp hw2")
parser.add_argument('--question_path',type=str, default="/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt")
parser.add_argument('--document_folder', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/documents')
parser.add_argument('--output_folder', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/')
parser.add_argument('--chunk_size', type=int, default=chunk_size)
parser.add_argument('--chunk_overlap', type=int, default=chunk_overlap)
parser.add_argument('--vecstore', type=str, default=vecstore)
parser.add_argument('--retrieve_k_docs', type=int, default=retrieve_k_docs)
parser.add_argument('--emb_model', type=str, default=embed_model)
parser.add_argument('--gen_model', type=str, default=llm)
parser.add_argument('--model_path', type=str, default=model_path)

args = parser.parse_args()
exp_name = f'closeBook_{args.gen_model}_{dt_string}'
question_file_path = args.question_path
question = load_random_question(question_file_path)
question_list = load_txt_file_all_rows(question_file_path)
print('lenght of question list:', len(question_list))
print(f'A random question: {question}')

def load_random_question(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return random.choice(questions).strip()

question_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt"
question = load_random_question(question_file_path)
print(f'Question: {question}')

n_gpu_layers = -1  
n_batch = 512  
llm = LlamaCpp(
    model_path="/home/ubuntu/nlp-from-scratch-assignment-spring2024/model/llama-2-13b-chat.Q4_K_M.gguf",
    # model_path="/home/ubuntu/llama-2-13b-chat.Q3_K_S.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True for efficiency
    verbose=True,
)

def ask_model(question):
    formatted_question = f"Question: {question}\nAnswer:"
    response = llm.invoke(formatted_question)
    parser = StrOutputParser() 
    answer = parser.parse(response)
    return answer

answer = ask_model(question)
print(f"Answer: {answer}")


res = []
qa_log_file_path = args.output_folder + f'qa_log_{exp_name}.txt'
with open(qa_log_file_path, 'w') as qa_log_file:  # New code
    for q in tqdm(question_list):
        result = ask_model(q)
        res.append(result)
        clean_result = clean_output_answer(result)
        qa_log_file.write(f"Question: {q}\nAnswer: {clean_result}\n\n")  # New code

res = [clean_output_answer(r) for r in res]

with open(args.output_folder + f'system_output_{exp_name}.txt', 'w') as f:
    for item in res:
        f.write("%s\n" % item)