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
    return "\n\n".join(doc.page_content for doc in docs)

def clean_output_answer(text):
    return text.replace('\n', ' ').strip()

def custom_rag_prompt(context, question):
    formatted_input = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Here is some example Question and Answer, you should answer the question in a similar way:
    Question: Who is offering the Exploring Pittsburgh course in Spring 2024?
    Answer: Torello

    Question: For Fall 2023, When is Mini-1 Last Day of Classes?
    Answer: October 13, 2023

    Question: {question} 
    Context: {context} 
    Answer: """
    return formatted_input


def generate_answer(question, vectorstore, llm, retriever, fewshot=0,
                    retrieve_k_docs=4, verbose=False, verbose_log_path="/home/ubuntu/nlp-from-scratch-assignment-spring2024/src/data/test/retrieval_result.txt"):
    docs = vectorstore.similarity_search(question, k=retrieve_k_docs)
    if verbose:
        with open(verbose_log_path, "a") as f:
            f.write(f'Question: {question}\n')
            f.write(f"\tNumber of retrieved documents: {len(docs)}\n")
            for doc in docs:
                f.write(f"\tDocument Name: {doc.metadata['source']}.")
                f.write(f"\t\tDocument Content: {doc.page_content}\n")
            f.write('-'*300 + '\n')

    if fewshot == 0:
        prompt_template = """
You are an assistant for question-answering tasks. Based on the retrieved context, your goal is to provide the answer in the shortest form possible, focusing solely on the key information requested in the question. Avoid any elaboration, additional context, or restating the question. Think of your responses as if they were data entries rather than sentences.  

Now, based on the context provided below, what is the direct answer to the following question?

Question: {question} 
Context: {context} 

The direct answer to the question "{question}" is: 
"""
    elif fewshot == 1:
        print('1-shot learning.. ')
        prompt_template = \
"""
You are an assistant for question-answering tasks. Based on the retrieved context, your goal is to provide the answer in the shortest form possible, focusing solely on the key information requested in the question. Avoid any elaboration, additional context, or restating the question. Think of your responses as if they were data entries rather than sentences. 

Here are two example Questions and Answers, you should answer the question in a similar way:
Question: Who is offering the Exploring Pittsburgh course in Spring 2024?
Answer: Torello

Question: For Fall 2023, When is Mini-1 Last Day of Classes?
Answer: October 13, 2023

Now, based on the context provided below, what is the direct answer to the following question? Be sure to only output the answer. 

Context: {context}
Question: {question}  
Answer: """
    elif fewshot == 3:
        print('3-shot learning.. ')
        prompt_template = \
"""
You are an assistant for question-answering tasks. Based on the retrieved context, your goal is to provide the answer in the shortest form possible, focusing solely on the key information requested in the question. Avoid any elaboration, additional context, or restating the question. Think of your responses as if they were data entries rather than sentences. 

Here are three example Questions and Answers, you should answer the question in a similar way:
Question: Who is offering the Exploring Pittsburgh course in Spring 2024?
Answer: Torello

Question: For Fall 2023, When is Mini-1 Last Day of Classes?
Answer: October 13, 2023

Question: Where is the Diploma Ceremony for Heinz College?
Answer: Petersen Events Center, University of Pittsburgh

Now, based on the context provided below, what is the direct answer to the following question? Be sure to only output the answer. 

Context: {context}
Question: {question}  
Answer: """
    else:
        raise ValueError(f'fewshot value {fewshot} not defined')

    # rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt_custom = PromptTemplate.from_template(prompt_template)
  
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(question)
    print(result)

    return result