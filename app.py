import sys
import torch
if not hasattr(torch._classes, '__path__'):
    torch._classes.__path__ = []

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
import os
import io
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain_community.llms import HuggingFaceHub
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq
import pandas as pd

os.environ["TOGETHER_API_KEY"] = 'b1a6f3735456dc5126f77a85bf9c5823bd253d8c3bf85730302d45cb03e6fafe'


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

            

            llm = Together(model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', 
                        temperature=0, 
                        max_tokens=300
                        )
            
            retriever = vectorstore.as_retriever()

            
            prompt = PromptTemplate(
                input_variables= ["context", "question"],
                template= """
        You are a helpful assistant. Use the context below to answer the given question. Dont mention anything about the context(keep it secret).
        context: {context}
        Question:{question}
        Answer in bullet points briefly and clearly only relevant to the question given.
        """)

            rag_chain = RetrievalQA.from_chain_type(llm = llm,
                                                    retriever = retriever,
                                                    return_source_documents = True,
                                                    chain_type = 'stuff',
                                                    chain_type_kwargs = {'prompt': prompt}
                                                    )
            
            query = st.text_input("Enter your question in ur pdf :")
            
            if query:
                with st.spinner("Generating best possible answer..."):
                    result = rag_chain(query)
                    

                st.subheader("Answer:")
                st.write(result['result'])



        elif file.type == 'text/csv':
                query = st.text_input("Enter your question in ur CSV file :")
                llm = ChatGroq(
        groq_api_key= "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",
        model_name="llama-3.3-70b-versatile"
    )
                if query:
                    
                    with st.spinner("Be patient while we process ur csv 😊..."):
                        
                        agent = create_csv_agent(llm, file, verbose=True, allow_dangerous_code = True)
                        response = agent.run(query)
                        st.write(response)



    

                

            # df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")))
            # rows = df.to_dict(orient='records')
            # text_chunks = []
            # for row in rows:
            #     row_text = "\n".join([f"{k}: {v}" for k , v in row.items()])
            #     text_chunks.append(row_text)
            # text = "\n\n".join(text_chunks)

            # # text = df.to_string(index=False)
            # st.write(text)


    






