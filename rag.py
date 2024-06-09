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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


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

    # logger.info('MD Files: %s', md_files)
    # logger.info('PDF Files: %s', pdf_files)

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

    QUESTION = "what is a pod in kubernetes"
    # QUESTION = "What are some acceptable model evaluation techniques?"
    # This should return keys to documents
    # logger.info(list(mem_store.yield_keys()))
    # This should return chunks from Chroma Vector DB
    # section_segments = chrome_vectordb.similarity_search(QUESTION, k=10)
    # logger.info(section_segments)
    # This should get Main document
    parentdoc_retriever.search_kwargs = {"k": 10}
    # full_sections = parentdoc_retriever.get_relevant_documents(QUESTION)
    # logger.info(full_sections)

    # logger.info("Vector search returned %s segments while the parent retriever returned %s sections", len(
    #     section_segments), len(full_sections))

    llm = Ollama(model="llama3")

    LLM_PROMPT_TEMPLATE = """Answer the question based only on the following context. 
        If the context does not provide sufficient information to answer the question, politely indicate that you are unable to assist. 
        
        {context}
        
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(LLM_PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    # in the first step we retrieve the context and pass through the input question
    setup_and_retrieval = RunnableParallel(
        {"context": parentdoc_retriever, "question": RunnablePassthrough()}

    )

    # In the subsequent steps pass the context and question to the prompt, send the prompt to the llm and parse the output as a string
    chain = setup_and_retrieval | prompt | llm | output_parser

    response = chain.invoke(QUESTION)
    logger.info(response)
