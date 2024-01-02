from transformers import pipeline
import pandas as pd
import sys

file = "./pokemons.csv"

data = pd.read_csv(file)

# pipe = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq", device=0)
pipe = pipeline("table-question-answering", model="microsoft/tapex-large-finetuned-wtq", device=0)

query = sys.argv[1]

result = pipe(table=data, query=query)
print(f"\nQuestion: {query}\nResponse:{result['answer']}")