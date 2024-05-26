# sqllite3 Python library that provides a lightweight disk-based database that doesn’t require a separate server process
import sqlite3

stock_ticker_data = [
    {
        "symbol": "PRAA",
        "name": "PRA Group, Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "AMZN",
        "name": "Amazon.com, Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "TSLA",
        "name": "Tesla Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "PAAS",
        "name": "Pan American Silver Corp.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "PAAC",
        "name": "Proficient Alpha Acquisition Corp.",
        "currency": "USD",
        "stockExchange": "NasdaqCM",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "RYAAY",
        "name": "Ryanair Holdings plc",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "MPAA",
        "name": "Motorcar Parts of America, Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "STAA",
        "name": "STAAR Surgical Company",
        "currency": "USD",
        "stockExchange": "NasdaqGM",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "RBCAA",
        "name": "Republic Bancorp, Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "AABA",
        "name": "Altaba Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGS",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "AAXJ",
        "name": "iShares MSCI All Country Asia ex Japan ETF",
        "currency": "USD",
        "stockExchange": "NasdaqGM",
        "exchangeShortName": "NASDAQ"
    },
    {
        "symbol": "ZNWAA",
        "name": "Zion Oil & Gas, Inc.",
        "currency": "USD",
        "stockExchange": "NasdaqGM",
        "exchangeShortName": "NASDAQ"
    }
]


# Creating a function for Db connection


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except FileExistsError as e:
        print(e)

    return conn


# Creating a function for table creation
def run_sql(conn, sql_query):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(sql_query)
    except RuntimeError as e:
        print(e)


# Insert the data created in the previous step.
def insert_data(conn, data):
    for item in data:
        conn.execute("INSERT INTO stock_ticker (symbol, name, currency,stockExchange, exchangeShortName ) VALUES (?, ?, ?, ?,?)",
                     (item["symbol"], item["name"], item["currency"], item["stockExchange"], item["exchangeShortName"]))
    conn.commit()
    conn.close()
