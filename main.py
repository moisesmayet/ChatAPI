import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Answer(BaseModel):
    answer: str


class Question(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "ChatAPI by Moisés Mayet"}


@app.post('/question-answer')
async def get_answer(question: Question):
    if question.question != 'string':
        os.environ['OPENAI_API_KEY'] = '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'
        pdf = SimpleDirectoryReader('library').load_data()
        modelo = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'))
        service_context = ServiceContext.from_defaults(llm_predictor=modelo)
        index = GPTVectorStoreIndex.from_documents(pdf, service_context=service_context)
        query_question = question.question + 'Responde en español'
        answer = index.as_query_engine().query(query_question).response
    else:
        answer = 'Puede preguntarme y con gusto le daré una respuesta.'
    return Answer(answer=answer)
