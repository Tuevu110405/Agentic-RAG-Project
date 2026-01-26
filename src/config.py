import torch
from pathlib import Path

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = "google/embeddinggemma-300m"
    router_model = "Qwen/Qwen2.5-1.5B-Instruct"
    agent_model = "Qwen/Qwen3-4B"
    router_max_new_tokens = 128
    router_temperature = 0.1
    agent_max_new_tokens = 1024
    agent_temperature = 0.6
    agent_is_quantized = False
    article_path = "/content/drive/MyDrive/Project_Datacom/article.txt"
    DATA_DIR = BASE_DIR / "data" / "raw"
    study_path = DATA_DIR / "study.txt"
    csv_path = DATA_DIR / "student_scores.csv"
    article_path 

config = Config()