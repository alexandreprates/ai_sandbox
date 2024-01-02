# Ai Sandbox

Repository with code examples written during my studies on the technology:

## ./tqa

Example of using _Table Question Answer_ to generate responses based on the content of tables.

```bash
$ python tqa.py "What is the rarest type of pokemon?"
```

## ./tokenizers

Example of how Tokenizers work and the importance of using the correct tokenizer for the model

```bash
$ python tokenizer.py "How does a machine understand a sentence?"
```

```bash
$ python truncate_tokenizer.py "This is doomed to failure"
```

## ./llm_interrogate

Example of how to use an LLM to answer questions about the content of a web page

```bash
$ python example.py "https://blog.juntossomosmais.com.br/institucional/loja-virtual-4-anos/" "What is the name of the CEO of Juntos Somos Mais"
```

## ./vector_store

Continuing from the previous example, we divided it into two parts: reading the base document (recording the vectors in a database), enabling offline queries

```bash
$ python index.py "https://blog.juntossomosmais.com.br/institucional/loja-virtual-4-anos/"
```

```bash
$ python interrogate.py "What is the name of the CEO of Juntos Somos Mais"
```

## ./llm_service

Continuing in the same line as the previous example, now implementing a webservice, this way we can reuse the model loaded into memory:

**For this example, I used FastAPI as a study, I know there are specialized frameworks that provide better performance**

```bash
$ python index.py "https://blog.juntossomosmais.com.br/institucional/loja-virtual-4-anos/"
```

```bash
$ uvicorn server:app --host 0.0.0.0
```

```bash
curl -X 'GET' \
  'http://localhost:8000/interrogate?query=Who%20is%20the%20CEO%20of%20Juntos%20Somos%20Mais%3F' \
  -H 'accept: application/json'
```

_TIP: You can use the swagger generated by fastapi to test:_ http://localhost:8000/docs#/default/interrogate_interrogate_get

## ./docker

Attempt to run the examples in a container, but the size of the models and slowness in the responses made this approach impractical.

# Attention:

This is a repository intended for studies, do not take into consideration the use of best practices or frameworks, tips and suggestions are welcome ;)