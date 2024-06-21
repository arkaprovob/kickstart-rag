from langchain_community.document_loaders import WebBaseLoader

from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = WebBaseLoader("http://paulgraham.com/greatwork.html")

pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)

splits = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
)

question = "what should I work on?"
docs = vectordb.similarity_search(question, k=1)

for doc in docs:
    print(doc)
