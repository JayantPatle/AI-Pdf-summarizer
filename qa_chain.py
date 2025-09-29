# qa_chain.py

from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

def get_qa_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key="API_Key",
        model_name="gemma2-9b-it"  # working supported model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain
