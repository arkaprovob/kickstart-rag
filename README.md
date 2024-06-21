# Getting Started with Vector Database and RAG

## Introduction

This guide is designed to help beginners understand and get started with the retrieval part of RAG (Retrieval-Augmented Generation). I will walk you through the process of setting up your environment, creating a Vector Database, and using it for information retrieval, which is a key component of the RAG process. 

This is the first piece of the puzzle. In the next sections, I will show you how to use this Vector Database with a Language Model and feed the retrieved information to the LM as a context. This will allow the LM to generate answers guided by prompts, which is the second part of the RAG process.

Stay tuned for the upcoming sections where we will delve deeper into the integration of retrieval with generation in the RAG framework.


## Prerequisites

- Basic knowledge of Python programming.
- Python 3.11 installed on your system.
- PyCharm or any other Python IDE installed on your system.

## Installation and setup

Before we start, make sure to install the necessary packages. You can do this by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## How to run this project
to keep things simple, i have created a `main.py` file that contains the code to create a Vector Database and use it for information retrieval. to run this file, simply execute the following command in your terminal:

```bash
python main.py
````

## Understanding the Code

First, we have to import the necessary modules and classes that we will be using in this project:
```python
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```

Next, we will create a `WebBaseLoader` instance to load our document, in this case, a web page. We will load the document from the URL `http://paulgraham.com/greatwork.html` langchain provides a `WebBaseLoader` class that can be used to load documents from a web page. Here's how you can use it

```python
loader = WebBaseLoader("http://paulgraham.com/greatwork.html")
pages = loader.load()
```

Next, we will use a `CharacterTextSplitter` to split our documents into smaller chunks. This is done to fit the documents more easily into a Language Model as context, which is used for tasks like text generation or question answering.
```python
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)
```


Next we will use `HuggingFaceEmbeddings` for our embeddings and `Chroma` for our vector store, An embedding is required because it converts the text into a numerical representation that can be used for similarity search. Here's how we have used it:

```python
embedding = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
)
```



## Using RAG for Information Retrieval

Now that we have our vector database, we can use RAG to retrieve information. For example, if we want to find the top 1 documents related to a specific question, we can do:

```python
question = "what should I work on?"
docs = vectordb.similarity_search(question, k=1)

for doc in docs:
    print(doc)
```

## Understanding the Parameters

Let's break down the parameters used in the `main.py` file:

1. `WebBaseLoader`: This is a class from the `langchain_community.document_loaders` module. It is used to load documents from a web page. The parameter `"http://paulgraham.com/greatwork.html"` is the URL of the web page from which the documents are to be loaded.

2. `CharacterTextSplitter`: This is a class from the `langchain_text_splitters` module. It is used to split the loaded documents into smaller chunks. The parameters are:
   - `separator="\n"`: This is the character used to split the document. In this case, it splits the document at every newline character.
   - `chunk_size=1000`: This is the maximum size of each chunk. In this case, each chunk will have a maximum of 1000 characters.
   - `chunk_overlap=150`: This is the number of characters that will overlap between two consecutive chunks. In this case, each chunk will overlap the next one by 150 characters.
   - `length_function=len`: This is the function used to calculate the length of a chunk. In this case, it uses the built-in `len` function which returns the number of characters in a string.

3. `HuggingFaceEmbeddings`: This is a class from the `langchain_huggingface` module. It is used to create embeddings for the chunks of documents. By default, it doesn't take any parameters in this case, so it uses the default parameters defined in the `HuggingFaceEmbeddings` class, although you can pass the model name and other parameters if needed.

4. `Chroma`: This is a class from the `langchain.vectorstores` module. It is used to create a vector store from the document chunks and their embeddings. The parameters are:
   - `documents=splits`: This is the list of document chunks.
   - `embedding=embedding`: This is the embedding instance used to create embeddings for the document chunks.

5. `vectordb.similarity_search(question, k=1)`: This is a method of the `Chroma` class. It is used to find the top `k` documents that are most similar to the `question`. The parameters are:
   - `question`: This is the query for which to find similar documents.
   - `k=1`: This is the number of top similar documents to return. In this case, it returns the top 3 similar documents.

## Conclusion

Congratulations! You have successfully created a Vector Database and used RAG for information retrieval. Keep exploring and happy coding!

> Remember, this is a basic guide. Depending on the specifics of your project, you may need to modify the code or install additional packages and Always refer to the official documentation of the packages you are using for the most accurate and up-to-date information. For more details, you can visit the official pages of [Langchain](https://www.langchain.com), [Chroma DB](https://docs.trychroma.com), and [HuggingFace](https://huggingface.co).