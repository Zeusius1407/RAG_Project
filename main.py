import PyPDF2
import google.generativeai as genai
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import transformers
import chromadb
import numpy as np


embed_model = SentenceTransformer("intfloat/e5-base-v2")
#genai.configure(api_key = os.environ["API_KEY_HERE"])
PDF_PATH = "./dataset/Inductions Catalogue.pdf"

client = chromadb.PersistentClient(path="./RAG_db")


def chunk_pdf(pdf_path: str, chunk_size=30):
    chunks = []
    with open(pdf_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file)
        current_chunk = ""
        
        for page in pdf.pages:
            text = page.extract_text()
            words = text.split()
            
            for word in words:
                current_chunk += word + " "
                if len(current_chunk.split()) >= chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
    
    return chunks

def embed(chunks, embed_model):
    embeddings = embed_model.encode(chunks, show_progress_bar = True)
    shape = embeddings.shape
    print(shape)
    return embeddings 

def create_vector_store(embeddings, chunks, client):
    collection = client.get_or_create_collection(name="RAG_db")
    collection.add(
        embeddings = embeddings,
        documents = chunks,
        ids = [f"chunk{i}" for i in range(1, len(chunks)+1)]
    )
    print("Added embeddings successfully!")

def retrieve_relevant_chunks(query, embedding_model, client, top_k=20):
    collection = client.get_collection(name = "RAG_db")
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results = top_k
    )
    relevant_chunks = results['documents'][0]
    return relevant_chunks

chunks = chunk_pdf(pdf_path=PDF_PATH, chunk_size=30)
embeddings = embed(chunks=chunks, embed_model=embed_model)
create_vector_store(embeddings=embeddings, chunks=chunks, client=client)
query = input("Enter your query: ")
retrieved_relevant_chunks = retrieve_relevant_chunks(query=query, embedding_model=embed_model, client=client)
