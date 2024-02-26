from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import random

def load_random_question(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return random.choice(questions).strip()

def load_questions(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return [q.strip() for q in questions]

def format_docs(docs):
    for doc in docs:
        print(f"Document Name: {doc.metadata['source']}")
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, vectorstore, llm, verbose=False):
    docs = vectorstore.similarity_search(question)
    if verbose:
        print(f'Question: {question}')
        print(f"\tNumber of retrieved documents: {len(docs)}")
        for doc in docs:
            print(f"\tDocument Name: {doc.metadata['source']}")

    rag_prompt = hub.pull("rlm/rag-prompt")

    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(question)

    return result