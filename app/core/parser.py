from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import Document

class DataParser(ABC):
    """
    Abstract base class for document parsers.

    This class defines the interface that all document parsers must implement.
    """

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """
        Parse a single file and return a list of LlamaIndex Document objects.

        Args:
            file_path (str): The path to the file to be parsed.

        Returns:
            List[Document]: A list of parsed documents.
        """
        pass

class UnstructuredParser(DataParser):
    """
    Concrete parser implementation using the unstructured library.

    This parser uses unstructured.io to handle complex documents with tables
    and sophisticated layouts.
    """

    def __init__(self, strategy: str = "hi_res"):
        """
        Initialize the UnstructuredParser.

        Args:
            strategy (str): The partitioning strategy to use. Defaults to "hi_res"
                           for high-resolution table and layout handling.
        """
        self.strategy = strategy

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse a file using unstructured library.

        Args:
            file_path (str): The path to the file to be parsed.

        Returns:
            List[Document]: A list of parsed LlamaIndex Document objects.
        """
        from unstructured.partition.auto import partition

        # Use unstructured to partition the document
        elements = partition(filename=file_path, strategy=self.strategy)

        # Convert unstructured elements to LlamaIndex Documents
        documents = []
        for element in elements:
            # Create a Document from each element
            doc = Document(
                text=str(element),
                metadata={
                    "file_path": file_path,
                    "element_type": type(element).__name__,
                    "source": "unstructured",
                }
            )
            documents.append(doc)

        return documents

class ParserFactory:
    """
    Factory class for creating document parsers.

    This factory allows switching between different parser implementations
    based on configuration.
    """

    @staticmethod
    def get_parser(parser_config: dict) -> DataParser:
        """
        Create and return a parser instance based on configuration.

        Args:
            parser_config (dict): Dictionary containing parser configuration.
                                Expected keys: 'type' and parser-specific config.

        Returns:
            DataParser: An instance of the requested parser type.

        Raises:
            ValueError: If the requested parser type is not supported.
        """
        parser_type = parser_config.get("type", "unstructured")

        if parser_type == "unstructured":
            # Get unstructured-specific config
            unstructured_config = parser_config.get("unstructured", {})
            strategy = unstructured_config.get("strategy", "hi_res")

            return UnstructuredParser(strategy=strategy)

        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")
