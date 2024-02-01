import torch
import numpy as np
from typing import List, Union
from websearch import WebSearch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders.async_html import AsyncHtmlLoader
from langchain.document_transformers.beautiful_soup_transformer import BeautifulSoupTransformer
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableConfig
from langchain_core.documents.base import Document
from langchain_core.pydantic_v1 import BaseModel
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


device = "cuda" if torch.cuda.is_available() else "cpu"
custom_prompt_template = """
Search the following documents to find the answer for the question: {context}

Provide a concise and well-structured answer, citing the documents you used.
The answer should be formatted as:

Question: {question}
Answer: 
"""

embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs= {'device': device})


def run_web_search(query: str):
    websearch = WebSearch()
    
    loader = AsyncHtmlLoader(websearch.search(query))
    html = loader.load()

    soup = BeautifulSoupTransformer()
    documents = soup.transform_documents(html, tags_to_extract=['p'])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents=documents)

    return texts


def find_similar_sentences(
    sentences: List[Document],
    query: str = 'Who is Furina From Genshin Impact?',
    embedding_model: BaseModel = embedding_model,
    top_k: int=5
):
    inputs = [query] + [sentence.page_content for sentence in sentences]
    outputs = embedding_model.embed_documents(inputs)

    query_embedding = np.array(outputs[0])
    sentences_embeddings = np.array(outputs[1:])

    distances_from_query = list(
        map(
            lambda index, embed: 
            {
                'distance': np.dot(query_embedding, embed),
                'index': index
            },
            range(len(sentences_embeddings)),
            sentences_embeddings
        )
    )
    distances_from_query = sorted(distances_from_query, key=lambda x: x['distance'], reverse=True)[:top_k]

    return [sentences[distance['index']] for distance in distances_from_query]


def load_llm():
    llm = CTransformers(
        model="model/llama-2-7b-chat.Q6_K.gguf",
        model_type="llama",
        max_new_tokens=1024,
        context_length=4096,
        temperature=0.5,
        device=device,
        stop=["Question:", "\n"]
    )
    return llm


def runnable():
    # Declare retriever chain
    # Here we are setting up a sequence of actions to retrieve information. This involves two steps: running a web search and getting the top results.
    retriever = RunnableSequence(
        RunnableLambda(run_web_search),
        RunnableLambda(find_similar_sentences)
    )

    # Declare model chain
    # We are loading a language model (LLM) to work with the retrieved information.
    llm = load_llm()

    # Declare prompt chain
    # We are defining a template for generating prompts, which will include variables 'context' and 'question'.
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    # Combine model with prompt chain
    # We are creating a chain that combines the language model (LLM) with the prompt template to process documents.
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Combine it's all together
    # Finally, we are creating a retrieval chain that incorporates both the retriever chain (for fetching information) and the chain for processing documents.
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain
