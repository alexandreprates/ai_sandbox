from transformers import AutoTokenizer
import sys

sentence = sys.argv[1]

def process(tokenizer_name:str, input:str):
    print(tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence)["input_ids"]
    print(f"Token Ids:\n{token_ids}")

    sentence_tokens = tokenizer.tokenize(sentence)
    print(f"Sentence tokens:\n{sentence_tokens}")

    decoded_string = tokenizer.decode(token_ids)
    print(f"Reconstructed string:\n{decoded_string}\n\n")


process("dslim/bert-base-NER", sentence)
process("Jean-Baptiste/roberta-large-ner-english", sentence)
process("gpt2", sentence)
process("almaghrabima/NER-7CAT-llama-2-7b", sentence)
