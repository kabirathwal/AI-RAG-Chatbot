import os
import signal
import sys
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings

# ADD PERSONAL API KEY:
GEMINI_API_KEY = ""

# Embeddings: Huggingface
# Vector store: ChromaDB
# LLM: Google Gemini pro 

def signal_handler(sig, frame):
    print('\n program has ended.')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"',"").replace("\n", " ")
    prompt = ("""
    You are a helpful and informative bot that answers questions using the given documents from the reference context included below. \
    Respond in a complete sentence, be comprehensive, and include all relevant background information. \
    If context is irrelevant to the answer, ignore it. \
    Please also provide the filename and page number the answer was found in underneath.
              QUESTION: '{query}'
              CONTEXT: '{context}'

              ANSWER:
             """).format(query=query, context=context)
    return prompt

def get_relevant_context_from_db(query):
    context = ""
    embeddings_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device' : 'cpu'})
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embeddings_function)
    search_results = vector_db.similarity_search(query, k=4)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer =model.generate_content(prompt)
    return answer.text

welcome_text = generate_answer("Introduce yourself quickly")
print(welcome_text)

while True:
    print("-------------------------------------")
    # print("Ask question:")
    query = input("Ask me a question:\n")
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    answer = generate_answer(prompt=prompt)
    print("\nAnswer:")
    print(answer)
