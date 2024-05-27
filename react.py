import logging
import sys
from util import get_stock_ticker, get_stock_price, get_recent_stock_news, get_financial_statements, stock_news_search
from chains import get_sql_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_stock(llm, query, db_chain):
    """Construct a chain that will always invoke all of the tools and use the output to respond to the user query."""

    company_name, ticker = get_stock_ticker(query, db_chain)
    logger.info("\n\n---> Query : \"%s\" \n", query)
    logger.info("\n---> Company Name : \"%s\" \n", company_name)
    logger.info("\n---> Ticker : \"%s\" \n", ticker)
    stock_data = get_stock_price(ticker, history=10)
    stock_financials = get_financial_statements(ticker)
    stock_news = get_recent_stock_news(company_name)

    analysis_template = """
            Given detail stock analysis, Use the available data and provide investment recommendation. 
            The user is fully aware about the investment risk, do not include any kind of warning like 
            'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer. 
            User question: {query}. You have the following information available about {company_name}. 
            Write (5-8) point investment analysis to answer user query, At the end conclude with proper explanation. 
            Try to Give positives and negatives: 
            <stock_price>
                {stock_data}
            </stock_price> 
            <stock_financials>
                {stock_financials}
            <stock_financials>
            <stock_news>
                {stock_news}
            </stock_news>
            Provide an analysis only base on the information provided above. Do not use any other external information or your own knowledge as the basis for the analysis.
    """

    analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
    analysis_chain = analysis_prompt | llm | StrOutputParser()

    analysis = analysis_chain.invoke(
        {
            "query": query,
            "company_name": company_name,
            "stock_data": stock_data,
            "stock_financials": stock_financials,
            "stock_news": stock_news,
        }
    )

    return analysis


if __name__ == "__main__":
    # llm = Ollama(model="llama3:70b")
    llm = Ollama(model="llama3")
    sql_chain = get_sql_chain(llm)

    analyze_result = analyze_stock(
        llm, "Is Tesla a good investment choice right now?", sql_chain)
    logging.info(analyze_result)

    # company_name, company_ticker = get_stock_ticker(
    #     "What is the main business of Amazon?", sql_chain)
    # logging.info("Company Name: %s", company_name)
    # logging.info("Company Ticker: %s", company_ticker)

    # logging.info("-------------------------------")
    # logging.info(get_stock_price("AMZN"))
    # logging.info("-------------------------------")
    # logging.info(get_recent_stock_news("Amazon"))
    # logging.info("-------------------------------")
    # logging.info(get_financial_statements("AMZN"))
    # logging.info("-------------------------------")
    # logging.info(stock_news_search("Amazon"))
