import yaml
from dotenv import load_dotenv
import os

load_dotenv()

def save_config(config: dict, config_path="config.yaml"):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace placeholders with environment variables
    if config.get("llm", {}).get("api_key") == "sk-...":
        config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")
        
    return config
