from src.config import config
from sentence_transformers import SentenceTransformer
from src.rag.vectordb import VectorDB
from src.rag.loader import Loader
from src.models.llm_loader import get_router, get_agent, get_cloud_model
from src.agents.main_agent import Main
from src.interface import ui
from src.models.masked_chinese_router import get_masked_chinese_router

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
if config.is_chinese_mask_router == True:
    llm_router = get_masked_chinese_router()
else:
    llm_router = get_router()
llm_agent = get_agent()
llm_cloud = get_cloud_model()

#initialize main flow 
app = Main(llm_mini = llm_router, llm_large = llm_agent, vector_db = vector_db, embed_model=embed_model, llm_cloud = llm_cloud)

ui = ui(app)
ui.launch(share=True, debug=True, inbrowser=True)