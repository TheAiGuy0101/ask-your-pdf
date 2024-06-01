import os
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
#from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from streamlit_chat import message
import json

if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []

st.set_page_config(layout="wide")
#col1, col2 = st.columns([1,2])

def send_click():
    if st.session_state.user != '':
        prompt = st.session_state.user
        if prompt:
          docs = knowledge_base.similarity_search(prompt)
       # llm = OpenAI()
        llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=  0.7, max_tokens = 1000,openai_api_key=os.environ.get("OPENAI_API_KEY"))
        chain = load_qa_chain(llm, chain_type="stuff")
        
        #llm_chain = jd_enhancer_template | llm
        with get_openai_callback() as cb:
              response = chain.invoke({"input_documents": docs, "question" :prompt})
              #response = chain.run(input_documents=docs, question=prompt)
        st.session_state.prompts.append(prompt)
        #data = json.loads(response)
        #text = data["output_text"]
        #st.write(data)
        #st.write(response["output_text"])
        st.session_state.responses.append(response["output_text"])
        #return response["output_text"]


load_dotenv()
# Left column: Upload PDF text
st.header("Upload PDF Text")
st.header("Ask your PDF 💬")

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
if pdf is not None:
  pdf_reader = PdfReader(pdf)

  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()

  #t1=f"""<font color='black'>{text}</fon>"""
  #with col2:
  #    html(t1, height=400, scrolling=True)
  

  # split into chunks
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(text)

  # create embeddings
  embeddings = OpenAIEmbeddings()
  knowledge_base = FAISS.from_texts(chunks, embeddings)

  # show user input 
  st.text_input("Ask a question about your PDF:", key="user")
  st.button("Send", on_click=send_click)

   # col1.write(response)
  if st.session_state.prompts:
    for i in range(len(st.session_state.responses)-1, -1, -1):
        message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)
        message(st.session_state.responses[i], key=str(i))
        
       