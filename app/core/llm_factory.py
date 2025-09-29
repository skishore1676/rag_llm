from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

def create_llm(config: dict, llm_type: str = None):
    """
    Create and return a configured LLM instance based on the provided config.

    Args:
        config (dict): The application configuration dictionary.
        llm_type (str, optional): Override the LLM type ('openai' or 'ollama'). Defaults to config.

    Returns:
        The configured LlamaIndex LLM object (OpenAI or Ollama).
    """
    current_llm_type = llm_type if llm_type else config["llm"]["type"]

    if current_llm_type == "ollama":
        llm = Ollama(
            model=config["llm"]["ollama_model"],
            base_url=config["llm"]["ollama_base_url"],
            temperature=config["llm"]["temperature"],
            request_timeout=120.0,  # Set a longer timeout (in seconds)
        )
    else:  # Default to openai
        llm = OpenAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            api_key=config["llm"]["api_key"],
        )

    return llm
