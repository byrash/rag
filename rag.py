""" Local RAG """
import logging
import sys
import os
import glob
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.llms import Ollama


from langchain_chroma import Chroma
from chromadb.config import (Settings)

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Constants
OLLAMA_LLM3_MODEL = "llama3"


def load_files(folder_path):
    md_files = []
    pdf_files = []

    # files = os.listdir(folder_path)
    files = glob.glob(folder_path+"/**/*.*", recursive=True)

    for file in files:
        if file.endswith('.md'):
            md_files.append(file)
        elif file.endswith('.pdf'):
            pdf_files.append(file)

    logger.info('MD Files: %s', md_files)
    logger.info('PDF Files: %s', pdf_files)

    loaders = [TextLoader(os.path.join(folder_path, file))
               for file in md_files]
    loaders.extend([PyPDFLoader(os.path.join(folder_path, file))
                    for file in pdf_files])
    tmp_docs = []
    for loader in loaders:
        tmp_docs.extend(loader.load())

    return tmp_docs


if __name__ == "__main__":
    # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/
    chrome_vectordb = Chroma(
        collection_name="documents", embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        client_settings=Settings(allow_reset=False, anonymized_telemetry=False,
                                 persist_directory="./chromadb/data")
    )
    # chrome_vectordb = Chroma.from_documents(
    #     documents=docs, embedding=OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./chromadb/data")
    # chrome_vectordb.persist()

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # Parent docs are being saved in memory
    mem_store = InMemoryStore()

    parentdoc_retriever = ParentDocumentRetriever(
        vectorstore=chrome_vectordb,
        docstore=mem_store,
        child_splitter=child_splitter,
        # parent_splitter=parent_splitter
    )
    # logger.info(docs)

    parentdoc_retriever.add_documents(load_files('.'), ids=None)

    # This should return keys to documents
    # logger.info(list(mem_store.yield_keys()))
    # This should return chunks from Chroma Vector DB
    logger.info(chrome_vectordb.similarity_search(
        "what is a pod in kubernetes"))
    # This should get Main document
    parentdoc_retriever.search_kwargs = {"k": 2}
    logger.info(parentdoc_retriever.invoke(
        "what is a pod in kubernetes"))

    llm = Ollama(model="llama3")
