import chainlit as cl
import torch
from langchain.llms.ctransformers import CTransformers
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA


custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up
an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer: 
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Question: {question}\n"),
        ("ai", "Answer: ")
    ]
)
DB_FAISS_PATH = "vectorstores/db_faiss"
ENABLE_VS = False

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    return prompt

def load_llm():
    llm = CTransformers(
        model="model/llama-2-7b-chat.Q6_K.gguf",
        model_type="llama",
        max_new_tokens=100,
        temperature=0.5,
        device=device,
        stop=["Question:", "\n"]
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs= {'device': device})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs= {'prompt': qa_prompt}
    )

    return qa

    
## Chainlit ##
@cl.on_chat_start
async def start():
    if ENABLE_VS:
        model = load_llm()
        runnable = prompt | model | StrOutputParser()
        cl.user_session.set("runnable", runnable)
    else:
        runnable = qa_bot()
        cl.user_session.set("runnable", runnable)


@cl.on_message
async def main(message: cl.Message):
    runnable: Runnable = cl.user_session.get("runnable")

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
    )
    cb.answer_reached = True

    msg = cl.Message(
        content=""
    )
    msg.streaming = True

    async for token in runnable.astream(
        input={"question": message.content} if ENABLE_VS else {"query": message.content},
        config=RunnableConfig(callbacks=[cb])
    ):
        if ENABLE_VS:
            await msg.stream_token(token)
        else:
            await msg.stream_token(token['result'])