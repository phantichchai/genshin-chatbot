import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
<<<<<<< HEAD
from chain import runnable
=======
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
>>>>>>> main


prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Question: {question}\n"),
        ("ai", "Answer: ")
    ]
)

<<<<<<< HEAD
=======
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
        max_new_tokens=1024,
        context_length=4096,
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
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs= {'prompt': qa_prompt}
    )

    return qa

    
>>>>>>> main
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