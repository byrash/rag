""" Local RAG """
import json
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
from operator import itemgetter
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate


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

    # QUESTION = "what is a pod in kubernetes"
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

    # LLM_PROMPT_TEMPLATE = """Answer the question based only on the following context.
    #     If the context does not provide sufficient information to answer the question, politely indicate that you are unable to assist.

    #     {context}

    #     Question: {question}
    #     """
    # prompt = ChatPromptTemplate.from_template(LLM_PROMPT_TEMPLATE)
    # output_parser = StrOutputParser()

    # # in the first step we retrieve the context and pass through the input question
    # setup_and_retrieval = RunnableParallel(
    #     {"context": parentdoc_retriever, "question": RunnablePassthrough()}

    # )

    # # In the subsequent steps pass the context and question to the prompt, send the prompt to the llm and parse the output as a string
    # chain = setup_and_retrieval | prompt | llm | output_parser

    # response = chain.invoke(QUESTION)
    # logger.info(response)

    # template to rephrase the question
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Skip any preamble or summarization, simply generate the rephrased question.

    {chat_history}
    
    Follow Up Input: {question}
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    # template to generate a response
    answer_template = """You are a foremost expert in model risk and model governance. Your job is to advise users on the best practices and guidelines in these areas.
    The context below provides relevant information to answer the question. You must use this context to provide a detailed and accurate response.
    Only answer questions related to model risk and model governance, if a user asks a question about a different topic, politely decline.

    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

    # instantiate a blank memory buffer
    memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question")

    # First we add a step to load memory from the buffer to feed into the prompt
    loaded_memory = RunnablePassthrough.assign(chat_history=RunnableLambda(
        memory.load_memory_variables) | itemgetter("history"),)

    # Next we generate the standalone question
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(
                x["chat_history"], human_prefix="human", ai_prefix="assistant"
            ),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    }
    # Retrieve the documents using the generated question
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | parentdoc_retriever,
        "question": lambda x: x["standalone_question"],
    }
    # Construct the inputs for the final prompt with the formatted context docs
    # template and function to format context documents
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
        template="{page_content}")

    final_inputs = {
        "context": lambda x: [format_document(doc, DEFAULT_DOCUMENT_PROMPT) for doc in x["docs"]],
        "question": itemgetter("question"),
    }
    # Send the final prompt to the llm
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | llm,
        "docs": itemgetter("docs"),
    }
    # And now we put it all together!
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    logger.info(
        "Question:: What types of model risks should be documented? \n\n")
    inputs = {"question": "What types of model risks should be documented?"}
    result = final_chain.invoke(inputs)
    logger.info(result["answer"])
    logger.info("source documents:%s \n", json.dumps(
        [doc.metadata for doc in result["docs"]], indent=2))

    # after every conversation turn we update the conversation state in the memory buffer
    memory.save_context(inputs, {"answer": result["answer"]})

    logger.info(
        "Question:: Can you provide some examples for a fraud detection model? \n\n")
    inputs = {
        "question": "Can you provide some examples for a fraud detection model?"}
    result = final_chain.invoke(inputs)
    logger.info(result["answer"])
    logger.info("source documents: %s\n", json.dumps(
        [doc.metadata for doc in result["docs"]], indent=2))

    memory.save_context(inputs, {"answer": result["answer"]})

    logger.info("Question:: How about for a credit scoring model? \n\n")

    inputs = {"question": "How about for a credit scoring model?"}
    result = final_chain.invoke(inputs)
    logger.info(result["answer"])
    logger.info("source documents: %s\n", json.dumps(
        [doc.metadata for doc in result["docs"]], indent=2))

    memory.save_context(inputs, {"answer": result["answer"]})
