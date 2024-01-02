from transformers import AutoTokenizer
import sys

sentence = sys.argv[1]

tokenizer1 = AutoTokenizer.from_pretrained("gpt2")
token_ids = tokenizer1(sentence)["input_ids"]

tokenizer2 = AutoTokenizer.from_pretrained("almaghrabima/NER-7CAT-llama-2-7b")
decoded_string = tokenizer2.decode(token_ids)
print(f'"{decoded_string}"')
