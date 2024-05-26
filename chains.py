import logging
import sys
import json
import os
import sys
import re
from functools import partial
from db import create_connection, run_sql, insert_data, stock_ticker_data

from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from operator import itemgetter

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def setup_and_seed_db():
    # Create a Database Connection
    db_name = "stock_ticker_database.db"
    conn = create_connection(db_name)

    drop_table_sql = """DROP TABLE IF EXISTS stock_ticker;"""

    # Create Table Query
    create_table_sql = """CREATE TABLE stock_ticker (
        symbol text PRIMARY KEY,
        name text NOT NULL,
        currency text,
        stockExchange text, 
        exchangeShortName text
    );"""

    # Calling create table user defined function
    if conn is not None:
        # drop existing table
        run_sql(conn, drop_table_sql)

        # create projects table
        run_sql(conn, create_table_sql)

        # Insert Data & close connection
        insert_data(conn, stock_ticker_data)
    else:
        print("Error! cannot create the database connection.")


def extract_from_xml_tag(response: str, tag: str) -> str:
    """Extract the text from the specified XML tag in the response string."""

    tag_txt = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    if tag_txt:
        return tag_txt.group(1)
    else:
        return ""


if __name__ == "__main__":
    llm = Ollama(model="llama3")

    setup_and_seed_db()

    db = SQLDatabase.from_uri("sqlite:///stock_ticker_database.db")

    sql_template = """You are a ANSI SQL expert with access to a database with the following tables: 

    <schema>
    {table_info}
    </schema>

    Generate a SQL Query that retrieves the top {top_k} records. Do not explain the query, just output the SQL.
    Place the output into <sql>...</sql> tags.

    Question: {input}"""

    sql_prompt = ChatPromptTemplate.from_template(sql_template)

    query_writer_chain = create_sql_query_chain(
        llm, db, prompt=sql_prompt, k=100)

    extract_sql = partial(extract_from_xml_tag, tag="sql")

    query_writer_chain = query_writer_chain | RunnableLambda(
        extract_sql)  # extract the SQL query from the response

    execute_query = QuerySQLDataBaseTool(db=db)
    # let's test if the query works
    # execute_query.invoke({"query": "SELECT * FROM stock_ticker LIMIT 5;"})

    answer_prompt = ChatPromptTemplate.from_template("""Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    <question> {question} </question>
    <sql> 
    {query} 
    </sql>
    <result>
    {result}
    </result>

    Provide a response to the question. Do not provide any additional information, do not explain the SQL query, and do not say that the results came from a SQL query. 
    The user has no knowledge of the backend data systems and mention of SQL will confuse them.
    """)

    answer_chain = answer_prompt | llm | StrOutputParser()

    full_sql_chain = (

        RunnablePassthrough
        .assign(query=query_writer_chain)
        .assign(result=itemgetter("query") | execute_query)

        | answer_chain
    )

    print(full_sql_chain.invoke(
        {"question": "What is the ticker symbol for Tesla in stock ticker table?"}))

    print(full_sql_chain.invoke(
        {"question": "What is the name for ticker AMZN in table?"}))
