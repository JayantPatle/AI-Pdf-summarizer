# vector_store.py

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "Google_API_Key"

def create_vector_store_from_text(text):
    # 1. Split text into manageable chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    # 2. Load HuggingFace Embeddings (using MiniLM model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # 3. Create FAISS vectorstore from documents and embeddings
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore
