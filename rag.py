""" Local RAG """
import logging
import re
import sys
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup

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


if __name__ == "__main__":
    documents = scrape_url("https://open5gs.org/open5gs/docs/")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    chrome_vectordb = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory="./chromadb/data")
    chrome_vectordb.persist()

    query_vector = embeddings.embed_query("Tell me about Kubernetes")
    docs = chrome_vectordb.similarity_search_by_vector(query_vector, k=3)
    print(docs[0].page_content)
