import torch
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import config


def get_router():
    print("Loading router model...")
    router_name = config.router_model
    tokenizer = AutoTokenizer.from_pretrained(router_name)
    model = AutoModelForCausalLM.from_pretrained(
        router_name,
        device_map="auto",
        trust_remote_code = True,
        quantization_config = None
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.router_max_new_tokens,
        temperature = config.router_temperature,
        repetition_penalty = 1.1,
        return_full_text = False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def get_agent():
    print("Loading router model...")
    router_name = config.agent_model
    tokenizer = AutoTokenizer.from_pretrained(config.agent_model)
    bnb_config = None
    if config.agent_is_quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        router_name,
        device_map="auto",
        trust_remote_code = True,
        quantization_config = None
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.agent_max_new_tokens,
        temperature = config.agent_temperature,
        repetition_penalty = 1.2,
        return_full_text = False
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

