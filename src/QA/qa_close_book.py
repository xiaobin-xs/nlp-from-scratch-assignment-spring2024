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

n_gpu_layers = -1  
n_batch = 512  


def load_random_question(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return random.choice(questions).strip()

question_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt"
question = load_random_question(question_file_path)
print(f'Question: {question}')

llm = LlamaCpp(
    model_path="/home/ubuntu/llama-2-13b-chat.Q4_0.gguf",
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