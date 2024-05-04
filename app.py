import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

cohere_api_key=os.getenv('COHERE_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("Chat With Document")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
'''
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}

'''
)

prompt1=st.text_input("Enter Your Question from documents")

pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

def vector_embedding():

    if 'vectors' not in st.session_state:
        st.session_state.embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
        st.session_state.loader = PyPDFLoader(pdf_docs)
        st.session_state.docs = st.session_state.loader.load()
        st.session_statetext_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectores = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


if st.button("Process"):
    vector_embedding()
    st.write("Vector Store DB is ready")

import time

if prompt1:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time:", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-------------------------------")