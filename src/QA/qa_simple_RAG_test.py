from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
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
import random
import os, sys
import argparse
from tqdm import tqdm

from utils import load_random_question, load_txt_file_all_rows, generate_answer


# def load_random_question(file_path):
#     with open(file_path, "r") as file:
#         questions = file.readlines()
#     return random.choice(questions).strip()


parser= argparse.ArgumentParser(description="RAG QA system for Anlp hw2")
parser.add_argument('--question_path',type=str, default="/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt")
parser.add_argument('--document_folder', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/tmp_doc')
parser.add_argument('--output_path', type=str, default='/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/system_output.txt')
parser.add_argument('--model',type=str, default='llama')
parser.add_argument('--chunk_size',type=int, default=500)
parser.add_argument('--model_path',type=str, default='/home/ubuntu/llama-2-13b-chat.Q4_0.gguf')
args = parser.parse_args()

question_file_path = args.question_path
question = load_random_question(question_file_path)
question_list = load_txt_file_all_rows(question_file_path)
print('lenght of question list:', len(question_list))
print(f'A random question: {question}')


print('Loading documents...')
loader = DirectoryLoader(args.document_folder, glob="*.txt", recursive=True, show_progress=True, loader_cls=TextLoader)
documents = loader.load()
print(f'Total Document Size: {len(documents)}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)
# There is another choice of embedding is to use GPT4ALL: GPT4AllEmbeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

# docs = vectorstore.similarity_search(question)
# print(f"Number of retrieved documents: {len(docs)}")
# for doc in docs:
#     print(f"Document Name: {doc.metadata['source']}")

n_gpu_layers = -1  # Adjust based on your hardware
n_batch = 512     # Adjust based on your hardware and memory requirements


if args.model=="llama":
    llm = LlamaCpp(
    model_path=args.model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True for efficiency
    verbose=True,
)
elif args.model=="gpt4all":
    # Currently cannot be called due to OS version
    llm = GPT4All(
        model="/home/ubuntu/nlp-from-scratch-assignment-spring2024/model/gpt4all-13b-snoozy-q4_0.gguf",
        max_tokens=2048,
    )


# def format_docs(docs):
#     for doc in docs:
#         print(f"Document Name: {doc.metadata['source']}")
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_prompt = hub.pull("rlm/rag-prompt")

# retriever = vectorstore.as_retriever()
# qa_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | rag_prompt
#     | llm
#     | StrOutputParser()
# )

# result = qa_chain.invoke(question)
# print(result)


res = []
for q in tqdm(question_list):
    result = generate_answer(q, vectorstore, llm, verbose=True)
    res.append(result)


def clean_output_answer(text):
    return text.replace('\n', ' ').strip()

res = [clean_output_answer(r) for r in res]

# Save the result to a file
with open(args.output_path, 'w') as f:
    for item in res:
        f.write("%s\n" % item)