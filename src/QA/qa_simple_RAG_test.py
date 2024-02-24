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


def load_random_question(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return random.choice(questions).strip()

question_file_path = "/home/ubuntu/nlp-from-scratch-assignment-spring2024/data/test/questions.txt"
question = load_random_question(question_file_path)
print(f'Question: {question}')


loader = DirectoryLoader('/home/ubuntu/nlp-from-scratch-assignment-spring2024/tmp_doc')
documents = loader.load()
print(f'Total Document Size: {len(documents)}')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function)

docs = vectorstore.similarity_search(question)
print(f"Number of retrieved documents: {len(docs)}")
for doc in docs:
    print(f"Document Name: {doc.metadata['source']}")

n_gpu_layers = -1  # Adjust based on your hardware
n_batch = 512     # Adjust based on your hardware and memory requirements

llm = LlamaCpp(
    model_path="/home/ubuntu/llama-2-13b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True for efficiency
    verbose=True,
)

def format_docs(docs):
    for doc in docs:
        print(f"Document Name: {doc.metadata['source']}")
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = hub.pull("rlm/rag-prompt")

retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

result = qa_chain.invoke(question)
print(result)