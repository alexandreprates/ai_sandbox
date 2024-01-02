import sys
import torch
import chromadb
from transformers import BitsAndBytesConfig
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.readers import BeautifulSoupWebReader


print("Loading LLM model")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="stabilityai/stablelm-zephyr-3b",
    tokenizer_name="stabilityai/stablelm-zephyr-3b",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
    generate_kwargs={"temperature": 0.9, "do_sample": True},
    device_map="auto",
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:intfloat/multilingual-e5-small")

url = sys.argv[1]

print(f"Fetch Data of {url}")
documents = BeautifulSoupWebReader().load_data([url])

print("Indexing data")
db = chromadb.PersistentClient(path="./chroma-db")
chroma_collection = db.get_or_create_collection("jsm")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, service_context=service_context
)

print("Done")