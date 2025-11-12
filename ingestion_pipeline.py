import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

JSON_FILES_DIR = "./json_files"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_json_files():
    json_files = [f for f in os.listdir(JSON_FILES_DIR) if f.endswith(".jsonl")]
    data = []
    for file in json_files:
        print(f"Loading file: {file}")
        with open(os.path.join(JSON_FILES_DIR, file), "r") as f:
            for line in f:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} documents")
    return data


def create_documents(data):

    documents = []
    for item in data:

        document_text = f"Title: {item['title']}\n\n Text: {item['text']}"

        documents.append(Document(page_content=document_text, metadata={
            "id": item["id"],
            "url": item["url"],
            "title": item["title"],
        }))
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="wikipedia_chunks",
    )
    return vector_store

def main():
    data = load_json_files()
    documents = create_documents(data)
    print(f"Created {len(documents)} documents")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    vector_store = create_vector_store(chunks)
    print(f"Created vector store")

if __name__ == "__main__":
    main()
    