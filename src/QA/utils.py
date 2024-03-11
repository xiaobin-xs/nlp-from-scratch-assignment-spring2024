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

def format_docs(docs, retrieve_result_path='/home/ubuntu/nlp-from-scratch-assignment-spring2024/src/data/test/retrieval_result.txt'):
    # with open(retrieve_result_path, "a") as f:
    #     for doc in docs:
    #         f.write(f"\tDocument Name: {doc.page_content}\n")
    # for doc in docs:
    #     print(f"Document Name: {doc.metadata['source']}")
    return "\n\n".join(doc.page_content for doc in docs)

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


def generate_answer(question, vectorstore, llm, retrieve_k_docs=4, 
                    verbose=False, verbose_log_path="/home/ubuntu/nlp-from-scratch-assignment-spring2024/src/data/test/retrieval_result.txt"):
    docs = vectorstore.similarity_search(question, k=retrieve_k_docs)
    if verbose:
        with open(verbose_log_path, "a") as f:
            f.write(f'Question: {question}\n')
            f.write(f"\tNumber of retrieved documents: {len(docs)}\n")
            for doc in docs:
                f.write(f"\tDocument Name: {doc.metadata['source']}.")
                f.write(f"\t\tDocument Content: {doc.page_content}\n")
            f.write('-'*300 + '\n')


    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Here are two example Questions and Answers, you should answer the question in a similar way:
    Question: Who is offering the Exploring Pittsburgh course in Spring 2024?
    Answer: Torello

    Question: For Fall 2023, When is Mini-1 Last Day of Classes?
    Answer: October 13, 2023

    Question: {question} 
    Context: {context} 
    Answer: """

    # Does not work at all
    template = """
    You are a knowledgeable assistant specializing in question-answering tasks, with a focus on providing accurate and succinct responses based on provided context. When answering, draw directly from the given context to ensure accuracy. If the answer is not contained within the context, respond with "I don't know." Below are examples illustrating how to use context to answer questions accurately:

    Example 1:
    Question: Who is offering the Exploring Pittsburgh course in Spring 2024?
    Answer: Torello

    Example 2:
    Question: For Fall 2023, when is the Mini-1 last day of classes?
    Answer: October 13, 2023

    Given the above examples, please answer the following question using the provided context. Ensure your answer is clear and directly supported by the context.

    Question: {question}
    Context: {context}
    Answer:
    """

    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt_custom = PromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(question)

    return result