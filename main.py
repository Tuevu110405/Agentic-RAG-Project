from src.config import config
from sentence_transformers import SentenceTransformer
from src.rag.vectordb import VectorDB
from src.rag.loader import Loader
from src.models.llm_loader import get_router, get_agent
from src.agents.main_agent import Main
from src.interface import ui

# initialize vector store
embed_model = SentenceTransformer(config.embed_model)


vector_db = VectorDB(embed_model=embed_model)
loader = Loader(embed_model=embed_model)

# Load dữ liệu vào VectorDB
print("   -> Indexing Study Data...")
study_chunks = loader.semantic_load(config.study_path, semantic_threshold=0.5)
vector_db.add_collection("study", study_chunks)

print("   -> Indexing Article Data...")
article_chunks = loader.semantic_load(config.article_path, semantic_threshold=0.5)
vector_db.add_collection("article", article_chunks)

# initialize lm
llm_router = get_router()
llm_agent = get_agent()

#initialize main flow 
app = Main(llm_mini = llm_router, llm_large = llm_agent, vector_db = vector_db, embed_model=embed_model)

ui = ui(app)
ui.launch(share=True, debug=True, inbrowser=True)