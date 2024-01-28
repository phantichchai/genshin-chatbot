import torch
from langchain_core.documents.base import Document
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain.document_loaders.async_html import AsyncHtmlLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from vectorstore_argparser import VectorStoreArgParser
from typing import Dict, List

DB_FAISS_PATH = "vectorstores/db_faiss"
LORE = "/Lore"
COMPANION = "/Companion"
HEADERS_TO_SPLIT_ON = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
LAST_INDEX_OF_LORE = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def isInfo(metadata: Dict[str, str]):
    return len(metadata) != 0


def search(search_url: str):
    loader = AsyncHtmlLoader(search_url)
    documents = loader.load()
    text_splitter_html = HTMLHeaderTextSplitter(HEADERS_TO_SPLIT_ON)
    texts_html = text_splitter_html.split_text(documents[0].page_content)
    filter_documents = []
    for i in range(len(texts_html)):
        if isInfo(texts_html[i].metadata):
            filter_documents.append(texts_html[i])
    return filter_documents


def refined_document(base_search_url):
    urls = [base_search_url, base_search_url + LORE, base_search_url + COMPANION]
    refined_documents = []
    for url in urls:
        refined_documents += search(url)
    return refined_documents


def create_vector_store(args: VectorStoreArgParser):
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs= {'device': DEVICE})
    texts = refined_document(args.search_url)

    if args.save_vector_store:
        try:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(DB_FAISS_PATH)
            print("Success save vector store")
        except Exception as e:
            print(f"An error occurred: {e}")

    print(f"Length of texts: {len(texts)}")

if __name__ == "__main__":
    arg_parser = VectorStoreArgParser()
    arguments = arg_parser.parse_args()
    create_vector_store(arguments)
