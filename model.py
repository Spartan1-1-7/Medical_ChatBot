from langchain_core.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings`
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA 
import chainlit as cl

DB_FAISS_PATH='/media/arindam-shukla/Linux Storage/medical_chatbot/medical_bot/vectorstores/db_faiss'

custom_prompt_template = """ Use the following pieces of information to answer the users question. if you dont know the answer, please just day that you dont know the answer, dont try to make up an answer.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt():
    """prompt template for QA retrieval for each vector stores
    """
    prompt= PromptTemplate(
                template=custom_prompt_template,
                input_variables=['context','question'])
    return prompt

def load_llm():
    """load the llm model"""
    # llm = CTransformers(model_name='/media/arindam-shukla/Linux Storage/medical_chatbot/medical_bot/llama-2-7b-chat.ggmlv3.q8_0.bin',
    llm = CTransformers(model='/media/arindam-shukla/Linux Storage/medical_chatbot/medical_bot/llama-2-7b-chat.ggmlv3.q8_0.bin',
    model_type='llama',
    max_new_tokens=512,
    temperature=0.5)
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(llm=llm, 
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True, 
        chain_type_kwargs={'prompt':prompt})
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# chainlit

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content='starting the bot')
    await msg.send()
    msg.content = 'Hi, Welcome to the medical bot. what is your query?'
    await msg.update()
    cl.user.set('chain', chain)

@cl.on_message
async def main(message):
    chain = cl.user.get('chain')
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=['FINAL', 'ANSWER'])
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res['source_documents']

    if sources:
        answer += f'\nSources: {str(sources)}'
    else:
        answer += f'\n No Sources Found'

    await cl.Message(content=answer).send()