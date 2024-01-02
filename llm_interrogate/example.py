import sys
import torch
import time
from contextlib import contextmanager
from transformers import BitsAndBytesConfig
from llama_index.readers import BeautifulSoupWebReader
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, VectorStoreIndex

url = sys.argv[1]
question = sys.argv[2]

benchmarks = []

@contextmanager
def bench(name:str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        benchmarks.append({"name": name, "elapsed_time": elapsed_time})


print("Loading LLM model")
with bench("LLM Load"):
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
        generate_kwargs={"temperature": 0.8, "do_sample": True},
        device_map="auto",
    )

    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:intfloat/multilingual-e5-small")


print(f"Fetch Data of {url}")
with bench("Fetch data"):
    documents = BeautifulSoupWebReader().load_data([url])

print("Indexing data")
with bench("Index data"):
    vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = vector_index.as_query_engine(response_mode="compact")

print("Use LLM with context data")
with bench("LLM Response"):
    response = query_engine.query(question)

print(f"\n\nQuestion: {question}\nResponse: '{response.response.strip()}'")

print("\n\nBenchmarks:")
for b in benchmarks:
    print(f'  {b["name"]}\t{b["elapsed_time"]}')
