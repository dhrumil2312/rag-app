import os
from ingestion_pipeline import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate



LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.4

def initialize_qa_system():
    print("Initializing QA system!")

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name="wikipedia_chunks"
    )

    llm = Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    prompt_template = """ You are a knowledgeable Wikipedia expert assistant. 
    Anser the question based only on the provided wikipedia content. 
    If you cannot find the answer in the context, say "I don't have context!"

    Context: 
    {context}

    Question : {input}

    Provide a clear and concise answer!
    
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variable=["context", "input"]
    )

    qa_chain = create_stuff_documents_chain(
        llm, 
        prompt
    )
    retreival_chain =  create_retrieval_chain(
        vector_store.as_retriever(
            search_kwargs={"k" : 5}
        ),
        qa_chain
    )

    print("Rag system is initiallized successfully!!!!")
    return retreival_chain


def format_answer(result):
    print("Answer: ")
    print("-"*60)
    print(result['answer'])
    print("-"*60)

    if result.get('context'):
        print(" Sources:")
        for i, doc in enumerate(result['context']):
            doc_id = doc.metadata.get('id')
            doc_title = doc.metadata.get('title')
            url = doc.metadata.get('url')
            
            print(f"Source {i}")
            print(f"\t\tID: {doc_id}")
            print(f"\t\tTitle: {doc_title}")
            print(f"\t\tURL: {url}")
            



def main():
    print("*"*60)
    print("Wikipedia Q&A System with RAG")
    print("*"*60)

    print("Powered by Hugging Face Structured Wikipedia dataset Using Ollama + ChromaDB + LangChain")

    qa_chain = initialize_qa_system()

    print("Ask questions about wikipedia! (Type 'quit' to exit)")

    while True:

        question = input("Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!!")
            break

        print("\n\n Thinking \n\n")

        try:
            result = qa_chain.invoke({"input": question})
            format_answer(result)
        except Exception as e:
            print("Uh oh these is an error")
            raise e

        
        print()



if __name__ == "__main__":
    main()
