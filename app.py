import sys
import torch
if not hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []
import os

from dotenv import load_dotenv

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
import io
from langchain.chat_models import ChatOpenAI
# from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
# from huggingface_hub import InferenceClient
# from langchain_community.llms import HuggingFaceHub
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq
# from langchain_community.llms import OpenAI
# import pandas as pd

load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

st.set_page_config(page_title= "Ask your PDF")
st.header("Ask your File 📄")

file = st.file_uploader("Upload your file", type = ['pdf','csv'])


if file is not None:
    st.write(file.type)
    with st.spinner("Be patient while we process ur pdf 😊..."):
        text = ''' '''
        if file.type == 'application/pdf':
            pdf_reader = PdfReader(file)
            # print(pdf_reader.pages)
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        
            chunks = splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            # st.write(chunks)
            # st.write(documents)
            embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-V2" )
            vectorstore = FAISS.from_documents(documents,embedding)
            retriever = vectorstore.as_retriever()
            query = st.text_input("Enter your question in ur pdf :")

            if query:
                prompt = PromptTemplate(
                    input_variables= ["context", "question"],
                    template= """
You are a helpful assistant. Use the provided context to answer the Question below.
Don't mention the content explicitly(keep it a secret). Be concise and answer in bullet points if necessary.
Context: {context}
Question: {question}
Answer:

"""
                )
                llm = ChatOpenAI( model_name = "gpt-3.5-turbo",temperature =0.1, openai_api_key = openai_api_key)
                
                chain = RetrievalQA.from_chain_type(llm , 
                                                    chain_type="stuff",
                                                    retriever = retriever,
                                                     chain_type_kwargs = {"prompt": prompt},
                                                     
                                                     )
                docs = retriever.get_relevant_documents(query)
                with st.spinner("Generating best possible answer..."):

                
                    response = chain.run(query)
                    st.write(response)




        elif file.type == 'text/csv':
                query = st.text_input("Enter your question in ur CSV file :")
                llm = ChatGroq(
        groq_api_key= groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
                if query:
                    
                    with st.spinner("Be patient while we process ur csv 😊..."):
                        
                        agent = create_csv_agent(llm, file, verbose=True, allow_dangerous_code = True)
                        response = agent.run(query)
                        st.write(response)



    



    






