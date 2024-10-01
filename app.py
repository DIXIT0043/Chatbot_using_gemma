import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv('HF_TOKEN')
#for langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Please respond to the question asked'),
    ('user', 'Question:{question}')
])

st.title('Langchain with Gemma Model')
input_text = st.text_input('What question you have in mind')

# Call model
llm = Ollama(model='gemma2:2b')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))