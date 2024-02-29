from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import random

def load_random_question(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return random.choice(questions).strip()

def load_txt_file_all_rows(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return [q.strip() for q in questions]

def format_docs(docs):
    with open("/home/ubuntu/nlp-from-scratch-assignment-spring2024/src/data/test/retrieval_result.txt","a") as f:
        for doc in docs:
            f.write(f"\tDocument Name: {doc.page_content}\n")
    for doc in docs:
        print(f"Document Name: {doc.metadata['source']}")
    return "\n\n".join(doc.page_content for doc in docs)

def custom_rag_prompt(context, question):
    formatted_input = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Here is some example Question and Answer, you should answer the question in a similar way:
    Question: Who is offering the Advanced NLP course in Spring 2024?
    Answer: Graham Neubig

    Question: On what date are the Mid-Semester & Mini-1 grades due for 2023-2024 academic year?
    Answer: October 23, 2023

    Question: {question} 
    Context: {context} 
    Answer: """
    return formatted_input



def generate_answer(question, vectorstore, llm, verbose=False, verbose_log_path="/home/ubuntu/nlp-from-scratch-assignment-spring2024/src/data/test/retrieval_result.txt"):
    docs = vectorstore.similarity_search(question)
    if verbose:
        with open(verbose_log_path,"a") as f:
            f.write(f'Question: {question}\n')
            f.write(f"\tNumber of retrieved documents: {len(docs)}\n")
            for doc in docs:
                f.write(f"\tDocument Name: {doc.metadata['source']}\n")

        # print(f'Question: {question}')
        # print(f"\tNumber of retrieved documents: {len(docs)}")
        # for doc in docs:
        #     print(f"\tDocument Name: {doc.metadata['source']}")


    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Here are two example Questions and Answers, you should answer the question in a similar way:
    Question: Who is offering the Advanced NLP course in Spring 2024?
    Answer: Graham Neubig

    Question: On what date are the Mid-Semester & Mini-1 grades due for 2023-2024 academic year?
    Answer: October 23, 2023

    Context: {context} 
    Question: {question} 
    Answer: """

    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt_custom = PromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(question)

    return result