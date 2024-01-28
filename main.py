import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from chain import runnable


prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Question: {question}\n"),
        ("ai", "Answer: ")
    ]
)

## Chainlit ##
@cl.on_chat_start
async def start():
    _runnable = runnable()
    cl.user_session.set("runnable", _runnable)


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

    async for output in runnable.astream(
        input={'retrieve_documents': message.content, "question": message.content},
        config=RunnableConfig(callbacks=[cb])
    ):
        print(output)