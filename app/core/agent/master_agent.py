import logging
from fastapi import HTTPException
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from ..querier import query_index
from .pandas_agent import PandasAgent
from ..llm_factory import create_llm

logger = logging.getLogger(__name__)

class MasterAgent:
    """
    Master Agent that orchestrates between document querying and data analysis tools.
    """

    def __init__(self, config: dict, index_name: str, excel_path: str):
        """
        Initialize the MasterAgent.

        Args:
            config (dict): The application configuration dictionary.
            index_name (str): The name of the index for document querying.
            excel_path (str): The absolute path to the Excel file for data analysis.
        """
        self.config = config
        self.index_name = index_name
        self.excel_path = excel_path
        self.llm = create_llm(config)

        # Define the query document store tool
        def query_document_store(question: str) -> str:
            """
            Use this tool to ask questions and retrieve information from the indexed text documents,
            such as Monthly Business Reviews (MBRs) and case studies. It is best for qualitative questions,
            summaries, and finding specific text-based information.
            """
            response = query_index(question, index_name, config)
            return response.get("answer", "No answer found.")

        document_tool = FunctionTool.from_defaults(fn=query_document_store)

        # Define the query data analysis tool
        def query_data_analysis(question: str) -> str:
            """
            Use this tool for questions that require data analysis, calculations, filtering, or plotting
            on the provided Excel spreadsheet. It is best for quantitative questions about revenue, profit,
            scenario metrics, or any other numerical data in the spreadsheet.
            """
            try:
                pandas_agent = PandasAgent(excel_path, config)
                return pandas_agent.query(question)
            except Exception as e:
                return f"Error in data analysis: {str(e)}"

        data_tool = FunctionTool.from_defaults(fn=query_data_analysis)

        # Create the agent with tools
        self.agent = ReActAgent.from_tools(
            [document_tool, data_tool],
            llm=self.llm,
            verbose=True,
            system_prompt="You are a helpful assistant that can access both document stores and Excel worksheets. Use the appropriate tool based on the query."
        )

    def query(self, question: str) -> str:
        """
        Query the MasterAgent using natural language.

        Args:
            question (str): The user's question.

        Returns:
            str: The agent's response.
        """
        try:
            # Use the chat method which returns a ChatMessage, convert to string
            response = self.agent.chat(question)
            return str(response)
        except Exception as e:
            logger.error(f"An error occurred during agent query: {str(e)}")
            return "I was unable to answer that question due to an error."
