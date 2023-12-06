from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

search_url = "https://genshin-impact.fandom.com/wiki/Furina/Lore"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_store():
    loader = AsyncHtmlLoader(search_url)
    documents = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        documents, tags_to_extract=["p"]
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents=docs_transformed)
    
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs= {'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_store()