from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings, LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import GPT4All
import os, sys
import argparse
from tqdm import tqdm

from utils import load_random_question, load_txt_file_all_rows, generate_answer, clean_output_answer

from datetime import datetime

def clean_output_answer(text):
    return text.replace('\n', ' ').strip()


now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")

# def load_random_question(file_path):
#     with open(file_path, "r") as file:
#         questions = file.readlines()
#     return random.choice(questions).strip()

## default hyperparams
chunk_size = 500 # e.g. 300, 500, 1000
chunk_overlap = 0
vecstore = "faiss" # 'faiss' or 'chroma'
retrieve_k_docs = 5
embed_model = 'gpt4all' # 'gpt4all', 'llama', or 'sentence-transformer'
llm = "llama" # 'gpt4all' or 'llama'
if llm == 'llama':
    model_path = '../../model/llama-2-13b-chat.Q4_0.gguf'
elif llm == 'gpt4all':
    model_path = '../../model/gpt4all-13b-snoozy-q4_0.gguf'
else:
    raise ValueError(f'llm model {llm} not defined')


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

exp_name = f'{args.chunk_size}_{args.chunk_overlap}_{args.vecstore}_{args.emb_model}_{args.gen_model}_{dt_string}'


question_file_path = args.question_path
question = load_random_question(question_file_path)
question_list = load_txt_file_all_rows(question_file_path)
print('lenght of question list:', len(question_list))
print(f'A random question: {question}')


print('Loading documents...')
loader = DirectoryLoader(args.document_folder, glob="*.txt", recursive=False, show_progress=True, loader_cls=TextLoader)
documents = loader.load()
print(f'Total Document Size: {len(documents)}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
all_splits = text_splitter.split_documents(documents)

if args.emb_model == 'gpt4all':
    embedding_function = GPT4AllEmbeddings()
elif args.emb_model == 'llama':
    embedding_function = LlamaCppEmbeddings(model_path=args.model_path)
elif args.emb_model == 'sentence-transformer':
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
else:
    raise ValueError(f'embedding method {args.emb_model} not defined')

vecdb_path = args.output_folder + f'log/{args.vecstore}_{args.chunk_size}_{args.chunk_overlap}_{args.emb_model}.vecdb'
if args.vecstore == 'faiss':
    # vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_function)
    if os.path.exists(vecdb_path):
        print(f'Loading vectorstore from {vecdb_path} ...')
        vectorstore = FAISS.load_local(vecdb_path, embeddings=embedding_function)
    else:
        print(f'Creating vectorstore and saving to {vecdb_path} ...')
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=embedding_function)
        vectorstore.save_local(folder_path=vecdb_path)
elif args.vecstore == 'chroma':
    # vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)
    if os.path.exists(vecdb_path):
        print(f'Loading vectorstore from {vecdb_path} ...')
        vectorstore = Chroma(persist_directory=vecdb_path, embedding_function=embedding_function)
    else:
        print(f'Creating vectorstore and saving to {vecdb_path} ...')
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function, persist_directory=vecdb_path)
else:
    raise ValueError(f'vector store {args.vecstore} not defined') 

n_gpu_layers = -1  # Adjust based on your hardware
n_batch = 512     # Adjust based on your hardware and memory requirements


if args.gen_model=="llama":
    llm = LlamaCpp(
    model_path=args.model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # Use half-precision for key/value cache, MUST set to True for efficiency
    verbose=True,
)
elif args.gen_model=="gpt4all":
    # Currently cannot be called due to OS version
    llm = GPT4All(
        model=args.model_path,
        max_tokens=2048,
        device='nvidia',
        f16_kv=True,   # Use half-precision for key/value cache, MUST set to True for efficiency
        verbose=True,
    )
else:
    raise ValueError(f'llm model {args.gen_model} not defined')


qa_log_file_path = args.output_folder + f'qa_log_{exp_name}.txt'  # New code

<<<<<<< Updated upstream
res = []
with open(qa_log_file_path, 'w') as qa_log_file:  # New code
    for q in tqdm(question_list):
        result = generate_answer(q, vectorstore, llm, 
                                 retrieve_k_docs=args.retrieve_k_docs, 
                                 verbose=True,
                                 verbose_log_path=args.output_folder + f'log/retrieval_result_{exp_name}.txt')
        res.append(result)
        clean_result = clean_output_answer(result)
        qa_log_file.write(f"Question: {q}\nAnswer: {clean_result}\n\n")  # New code
=======
# def clean_output_answer(text):
#     return text.replace('\n', ' ').strip()
>>>>>>> Stashed changes

res = [clean_output_answer(r) for r in res]

# Save the result to a file (this part remains unchanged)
with open(args.output_folder + f'system_output_{exp_name}.txt', 'w') as f:
    for item in res:
        f.write("%s\n" % item)