""" Local RAG """
import logging
import re
import sys
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup as Soup
import chromadb
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)
from chromadb.config import (Settings)

# Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Constants
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
    docs_as_txt = list(map(lambda doc: doc.page_content, docs))
    return docs_as_txt


def generate_embeddings(texts):
    """Generates Embeedings"""
    logger.info("Generating embeedings for [%s] documents", len(texts))
    logger.debug("Generating embeedings for [%s]", texts)
    embeddings = OllamaEmbeddings(model="llama3")
    logger.debug("Type of %s is %s", texts[0], type(texts[0]))
    doc_result = embeddings.embed_documents(texts)
    logger.debug("Embeeding Result %s", doc_result)
    return doc_result


class CustomEmbeddingFunction(EmbeddingFunction):
    """Plugin to leverage Ollama Embeedings for a given text"""

    def __call__(self, texts: Documents) -> Embeddings:
        logger.info("[%s] Texts to embeed", len(texts))
        return generate_embeddings(texts)


def setup_db():
    """Sets up Chrome Disk Persistent Client & Creates a collection"""
    client = chromadb.Client(settings=Settings(chroma_db_impl="duckdb+parquet",
                                               allow_reset=False, anonymized_telemetry=False,
                                               persist_directory="./chromadb/data"))
    collection = client.get_or_create_collection(
        name="Shivaji_RAG_LLM", embedding_function=CustomEmbeddingFunction())
    return client, collection


def save_embeedings(client, collection, docs):
    """Generates Embeedings ( Chroma Client configuration ) & saves to DB """
    doc_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(docs)))
    collection.add(documents=docs, ids=doc_ids)
    client.persist()


def pure_ollama():
    """Pure Ollama Embeeding Generation"""
    embeddings = OllamaEmbeddings(model="llama3")
    doc_result = embeddings.embed_documents(scrape_url("https://google.com"))
    logger.info("%s", doc_result[0][:5])


def get_mock_documents():
    """Mock Documents"""
    return [
        "Mars, often called the 'Red Planet', has captured the imagination of scientists and space enthusiasts alike.",
        "The Hubble Space Telescope has provided us with breathtaking images of distant galaxies and nebulae.",
        "The concept of a black hole, where gravity is so strong that nothing can escape it, was first theorized by Albert Einstein's theory of general relativity.",
        "The Renaissance was a pivotal period in history that saw a flourishing of art, science, and culture in Europe.",
        "The Industrial Revolution marked a significant shift in human society, leading to urbanization and technological advancements.",
        "The ancient city of Rome was once the center of a powerful empire that spanned across three continents.",
        "Dolphins are known for their high intelligence and social behavior, often displaying playful interactions with humans.",
        "The chameleon is a remarkable creature that can change its skin color to blend into its surroundings or communicate with other chameleons.",
        "The migration of monarch butterflies spans thousands of miles and involves multiple generations to complete.",
        "Christopher Nolan's 'Inception' is a mind-bending movie that explores the boundaries of reality and dreams.",
        "The 'Lord of the Rings' trilogy, directed by Peter Jackson, brought J.R.R. Tolkien's epic fantasy world to life on the big screen.",
        "Pixar's 'Toy Story' was the first feature-length film entirely animated using computer-generated imagery (CGI).",
        "Superman, known for his incredible strength and ability to fly, is one of the most iconic superheroes in comic book history.",
        "Black Widow, portrayed by Scarlett Johansson, is a skilled spy and assassin in the Marvel Cinematic Universe.",
        "The character of Iron Man, played by Robert Downey Jr., kickstarted the immensely successful Marvel movie franchise in 2008."
    ]


if __name__ == "__main__":
    chromadb_client, chromadb_collection = setup_db()
    # documents = scrape_url("https://open5gs.org/open5gs/docs/")
    # # documents = get_mock_documents()
    # save_embeedings(chromadb_client, chromadb_collection, documents)

    query_result = chromadb_collection.query(
        query_texts="Elastic Kubernetes Service", n_results=3)
    logger.info("Query Result [%s]", query_result)
    result_documents = query_result["documents"][0]

    logger.info(" ----------- Results Start ----------- ")
    for doc in result_documents:
        logger.info("%s", doc)
    logger.info(" ----------- Results End ----------- ")
