import logging
import sys
from util import extract_docstring_info, construct_format_tool_prompt, CustomTools
from chains import get_sql_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import XMLAgentOutputParser

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# def analyze_stock(llm, query, db_chain):
#     """Construct a chain that will always invoke all of the tools and use the output to respond to the user query."""

#     company_name, ticker = get_stock_ticker(query, db_chain)
#     logger.info("\n\n---> Query : \"%s\" \n", query)
#     logger.info("\n---> Company Name : \"%s\" \n", company_name)
#     logger.info("\n---> Ticker : \"%s\" \n", ticker)
#     stock_data = get_stock_price(ticker, history=10)
#     stock_financials = get_financial_statements(ticker)
#     stock_news = get_recent_stock_news(company_name)

#     analysis_template = """
#             Given detail stock analysis, Use the available data and provide investment recommendation.
#             The user is fully aware about the investment risk, do not include any kind of warning like
#             'It is recommended to conduct further research and analysis or consult with a financial advisor before making an investment decision' in the answer.
#             User question: {query}. You have the following information available about {company_name}.
#             Write (5-8) point investment analysis to answer user query, At the end conclude with proper explanation.
#             Try to Give positives and negatives:
#             <stock_price>
#                 {stock_data}
#             </stock_price>
#             <stock_financials>
#                 {stock_financials}
#             <stock_financials>
#             <stock_news>
#                 {stock_news}
#             </stock_news>
#             Provide an analysis only base on the information provided above. Do not use any other external information or your own knowledge as the basis for the analysis.
#     """

#     analysis_prompt = ChatPromptTemplate.from_template(analysis_template)
#     analysis_chain = analysis_prompt | llm | StrOutputParser()

#     analysis = analysis_chain.invoke(
#         {
#             "query": query,
#             "company_name": company_name,
#             "stock_data": stock_data,
#             "stock_financials": stock_financials,
#             "stock_news": stock_news,
#         }
#     )

#     return analysis


if __name__ == "__main__":
    # llm = Ollama(model="llama3:70b")
    llm = Ollama(model="llama3")
    sql_chain = get_sql_chain(llm)
    custom_tools = CustomTools(db_chain=sql_chain, llm=llm)

    tools = [
        Tool(
            name="get_stock_ticker",
            func=custom_tools.get_stock_ticker,
            description="Get the company stock ticker",
        ),
        Tool(
            name="get_stock_price",
            func=custom_tools.get_stock_price,
            description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the the stock ticker to it "
        ),
        Tool(
            name="get_recent_stock_news",
            func=custom_tools.get_recent_stock_news,
            description="Use this to fetch recent news about stocks"
        ),

        Tool(
            name="get_financial_statements",
            func=custom_tools.get_financial_statements,
            description="Use this to get financial statement of the company. With the help of this data companys historic performance can be evaluaated. You should input stock ticker to it"
        )

    ]

    # 3 -
    agent_prompt_template = """
        You are a financial advisor who has access to a set of tools that can help answer questions about stocks and investments.

        You have access to the following tools:

        {tools}

        When you are done, respond with a final answer between <final_answer></final_answer>. 
        For example:

        <final_answer>Investment Analysis of Amazon.com, Inc. (AMZN)</final_answer>

        Make sure you make use of all the tools available to you and not bias your answer based on an output from a single tool or your own knowledge.

        Begin!

        Question: {question}
        """

    agent_prompt = ChatPromptTemplate.from_template(
        agent_prompt_template, partial_variables={"tools": tools})

    agent_chain = (
        {
            "question": lambda x: x["input"],
        }
        | agent_prompt
        | llm.bind(stop=["</final_answer>"])
        | XMLAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent_chain, tools=tools, verbose=True)
    agent_response = agent_executor.invoke(
        {"input": "Is Tesla a good investment choice right now?"})
    logger.info(agent_response["output"])

    # 2 -
    # analyze_result = analyze_stock(
    #     llm, "Is Tesla a good investment choice right now?", sql_chain)
    # logging.info(analyze_result)

    # 1 -
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
