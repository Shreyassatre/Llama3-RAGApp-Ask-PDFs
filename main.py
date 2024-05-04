import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import ollama
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import HuggingFaceHub
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv('HUGGINGFACEHUB_API_TOKEN')
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key=os.getenv('COHERE_API_KEY')


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
# llm = ChatOllama(model="llama2")
# llm = HuggingFaceHub(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="text-generation",
#     model_kwargs={
#         "max_new_tokens": 512,
#         "top_k": 30,
#         "temperature": 0.1,
#         "repetition_penalty": 1.03,
#     },
# )
# llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")
# embeddings=ollama.OllamaEmbeddings(model='nomic-embed-text')
# embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

history = []

prompt = ChatPromptTemplate.from_template(
'''
Answer the questions based on the provided context only.
Please provide the most accurate response based on the user provided question question
<context>
{context}
</context>
Questions: {input}

Note: Consider [history] also if len(history)>=2 in context to keep track of conversation.

'''
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def handle_userinput(user_question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({'input':user_question})
    history.append(user_question)
    history.append(response['answer'])

    for i, message in enumerate(history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    st.header("RAG App - Ask your PDFs  :books:")
    user_question=st.text_input("Enter Your Question from documents")
    if user_question:
        handle_userinput(user_question)

    user_question = ''

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                if 'vectorstore' not in st.session_state:
                    st.session_state.vectorstore = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()