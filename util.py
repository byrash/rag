import warnings
from datetime import timedelta
from langchain.tools import DuckDuckGoSearchRun
from bs4 import BeautifulSoup
import re
import requests
from pandas_datareader import data as pdr
from datetime import date
import yfinance as yf

yf.pdr_override()


warnings.filterwarnings("ignore")


def extract_from_xml_tag(response: str, tag: str) -> str:
    """Extract the text from the specified XML tag in the response string."""

    tag_txt = re.search(rf'<{tag}>(.*?)</{tag}>', response, re.DOTALL)
    if tag_txt:
        return tag_txt.group(1)
    else:
        return ""


def extract_docstring_info(docstring):
    """
    Extracts the description, parameters, and their types and descriptions from a Google style docstring.

    Args:
        docstring (str): The docstring to extract information from.

    Returns:
        dict: A dictionary containing the description, parameters, and their types and descriptions.
    """
    # Find the indices of "Args:" and "Returns:"
    if (docstring is None):
        return {"description": "Lambda", "params": []}

    args_index = docstring.index("Args:")

    returns_index = docstring.index("Returns:")

    # Extract the description
    function_description = docstring[:args_index].strip()

    # Extract the parameters
    param_string = docstring[args_index + 6: returns_index].strip()
    params = []
    for param in param_string.split("\n"):
        if param.strip() == "":
            continue
        param_name_type, description = param.strip().split(":")
        param_name, param_type = param_name_type.split("(")
        param_type = param_type[:-1]

        params.append(
            {
                "name": param_name.strip(),
                "type": param_type.strip(),
                "description": description.strip(),
            }
        )

    return {"description": function_description, "params": params}


def construct_format_parameters_prompt(parameters):
    constructed_prompt = "\n".join(
        f"<parameter>\n<name>{parameter['name']}</name>\n<type>{parameter['type']
                                                                }</type>\n<description>{parameter['description']}</description>\n</parameter>"
        for parameter in parameters
    )

    return constructed_prompt


def construct_format_tool_prompt(name, description, parameters):
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )
    return constructed_prompt


class CustomTools:
    def __init__(self, db_chain, llm) -> None:
        self.db_chain = db_chain
        self.llm = llm

    def get_stock_ticker(self, query):
        """
        Returns the ticker symbol and company name relevant to the given query.

        Args:
            query (str): The query to find the relevant ticker symbol and company name.
            db_chain (Chain): The database chain to use.

        Returns:
            tuple: A tuple containing the company name and ticker symbol.

        """

        response = self.db_chain.invoke(
            {
                "question": f"Return the ticker symbol and company name that is relevant to this query {query}. Place symbol into <symbol>...</symbol> tags and company name into <company>...</company> tags."
            }
        )
        symbol = extract_from_xml_tag(response, "symbol")
        company_name = extract_from_xml_tag(response, "company")

        return company_name, symbol

    def get_stock_price(self, ticker, history=10):
        """
        Returns the stock data for the specified ticker symbol.

        Args:
            ticker(str): The ticker symbol for which to fetch the stock data.
            history(int): The number of days of historical data to fetch. Defaults to 10.

        Returns:
            tuple: A tuple containing the stock data and a unique name.

        """

        today = date.today()
        start_date = today - timedelta(days=history)
        data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
        dataname = ticker + "_" + str(today)

        return data, dataname

    # Fetch top 5 google news for given company name

    def google_query(self, search_term):

        if "news" not in search_term:
            search_term = search_term + " stock news"
        url = f"https://www.google.com/search?q={search_term}"
        url = re.sub(r"\s", "+", url)
        return url

    def get_recent_stock_news(self, company_name):
        """
        Fetches and returns the top 5 recent news articles for a given company from Google News.

        Args:
            company_name (str): The name of the company to fetch news for.

        Returns:
            str: A string containing the top 5 recent news articles for the company.
        """

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        }

        g_query = self.google_query(company_name)
        res = requests.get(g_query, headers=headers).text
        soup = BeautifulSoup(res, "html.parser")
        news = []
        for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
            news.append(n.text)
        for n in soup.find_all("div", "IJl0Z"):
            news.append(n.text)

        if len(news) > 6:
            news = news[:4]
        else:
            news = news
        news_string = ""
        for i, n in enumerate(news):
            news_string += f"{i}. {n}\n"
        top5_news = "Recent News:\n\n" + news_string

        return top5_news

    def stock_news_search(self, company_name):
        search = DuckDuckGoSearchRun()
        return search("Stock news about " + company_name)

    # Get financial statements from Yahoo Finance

    def get_financial_statements(self, ticker):
        """
        Fetches and returns the balance sheet for a given company ticker from Yahoo Finance.

        Args:
            ticker (str): The ticker of the company to fetch the balance sheet for.

        Returns:
            str: A string representation of the company's balance sheet.
        """

        if "." in ticker:
            ticker = ticker.split(".")[0]
        else:
            ticker = ticker
        company = yf.Ticker(ticker)
        balance_sheet = company.balance_sheet
        if balance_sheet.shape[1] >= 3:
            # Only captures last 3 years of data
            balance_sheet = balance_sheet.iloc[:, :3]
        balance_sheet = balance_sheet.dropna(how="any")
        balance_sheet = balance_sheet.to_string()
        return balance_sheet
