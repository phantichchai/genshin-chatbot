import torch
from websearch import WebSearch
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders.async_html import AsyncHtmlLoader
from langchain.document_transformers.beautiful_soup_transformer import BeautifulSoupTransformer
from langchain.llms.ctransformers import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

device = "cuda" if torch.cuda.is_available() else "cpu"
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
"""


def run_web_search(query: str):
    websearch = WebSearch()
    
    loader = AsyncHtmlLoader(websearch.search(query))
    html = loader.load()

    soup = BeautifulSoupTransformer()
    documents = soup.transform_documents(html, tags_to_extract=['p'])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents=documents)

    return texts


def get_top(docs):
    return docs[:1]


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
        RunnableLambda(get_top)
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
