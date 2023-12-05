import chainlit as cl
import torch
from langchain.llms.ctransformers import CTransformers
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers.langchain import LangchainGenericProvider


device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Question: {question}\n"),
        ("ai", "Answer: ")
    ]
)
template = "Hello, {name}!"
inputs = {"name": "John"}

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

## Chainlit ##
@cl.on_chat_start
async def start():
    model = load_llm()
    runnable = prompt | model | StrOutputParser()
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
        input={"question": message.content},
        config=RunnableConfig(callbacks=[cb])
    ):
        await msg.stream_token(token)