""" Local RAG """
import logging
import re
import sys
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup
from langchain.document_loaders import PyPDFLoader
from pdf_utils import PyPDFOutlineParser
from langchain_community.llms import Ollama

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Constants
OLLAMA_LLM3_MODEL = "llama3"
PREFIXES_TO_IGNORE = ("javascript:", "mailto:", "#")
SUFFIXES_TO_IGNORE = (
    ".css",
    ".js",
    ".ico",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".csv",
    ".bz2",
    ".zip",
    ".epub",
)
SUFFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) +
                      r"[\#'\"]" for s in SUFFIXES_TO_IGNORE]) + ")"
)
PREFIXES_TO_IGNORE_REGEX = (
    "(?!" + "|".join([re.escape(s) for s in PREFIXES_TO_IGNORE]) + ")"
)


def scrape_url(url):
    """Scrapes a given URL and returns a vector of Texts"""
    logger.info("Loading URL [%s]", url)
    # scrape data from web
    docs = RecursiveUrlLoader(
        url,
        max_depth=4,
        extractor=lambda x: Soup(x, "html.parser").text,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True,
        # drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{
                PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
    ).load()

    # split text
    # this chunk_size and chunk_overlap effects to the prompt size
    # execeed promt size causes error `prompt size exceeds the context window size and cannot be processed`
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    # docs_as_txt = list(map(lambda doc: doc.page_content, docs))
    # return docs_as_txt
    return docs[:5]


def load_pdf():
    pdf_files = list(Path(".").glob("**/*.pdf"))
    section_chunks_local = []

    for pdf in pdf_files:
        loader = PyPDFLoader(file_path=pdf.as_posix())
        loader.parser = PyPDFOutlineParser()
        sections = loader.load()
        for sec in sections:
            sec.metadata.update({"file": pdf.name})

        section_chunks_local += sections
    return section_chunks_local


if __name__ == "__main__":
    section_chunks = load_pdf()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chrome_vectordb = Chroma.from_documents(
        documents=section_chunks, embedding=embeddings, persist_directory="./chromadb/data")
    chrome_vectordb.persist()

    llm = Ollama(model="llama3")
