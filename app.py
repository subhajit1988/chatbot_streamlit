import pandas as pd
import time
import openai
import sys
import tiktoken
import numpy as np
import warnings
import io
import toml
import threading
import random
import logging 
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from PyPDF2 import PdfReader
#from langchain.text_splitter import CharacterTextSplitter
from preprocess import *
from embeddings import *
#from main_functions import *
from app_functions_2dfs import *
import json
from streamlit_feedback import streamlit_feedback
import base64
import ast
from docx import Document

secrets ="secrets.toml"

# Configure logging at the start of your script
logging.basicConfig(level=logging.DEBUG, filename='app_debug.log', format='%(asctime)s - %(levelname)s - %(message)s')

#secrets
toml_dir = secrets
file_name = ""
with open(toml_dir, 'r') as f:
    config = toml.load(f)

# Set up OpenAI API credentials
# azure required env vars commented out 
openai.api_type = config['api_type']
openai.api_base = config['api_base']
openai.api_version = config['api_version']
openai.api_key = config['api_key']
# Set up engine name
engine_name = config['engine_name']

#Completion on knowlede store config
EMBEDDING_MODEL = config['EMBEDDING_MODEL']
#messages_ai = []

def define_dataframe_types(df):
    df.loc[:, 'heading'] = df['heading'].astype(str)
    df.loc[:, 'title'] = df['title'].astype(str)
    df.loc[:, 'tokens'] = df['tokens'].astype(int)
    df.loc[:, 'content'] = df['content'].astype(str)
    
def reset_messages_ai():
    global messages_ai
    # Reset messages_ai to its default state or an empty list/dictionary, depending on your implementation
    print("jhdgfgsdhfghsdghsdghgshydgsygdsyigyisgysdgsdsygdsyugdsyugds")
    messages_ai = []

def load_dfs(local_toml, idx_load = ['title','heading']):
    global data_sources
    global doc_embeddings 
    global code_embeddings 
    global df_code
    global df_doc
    global app_prompt_list
 
    df = pd.read_parquet(config[local_toml])

    df_code= df
    define_dataframe_types(df_code)

    code_embeddings = load_embeddings(df_code)
    df_code = df_code.set_index(idx_load)
    reset_messages_ai()
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

def load_cusrrent_dfs(file_name, idx_load = ['title','heading']):
    global data_sources
    global doc_embeddings 
    global code_embeddings 
    global df_code
    global df_doc
    global app_prompt_list
 
    df = pd.read_parquet('data_sources/'+file_name+'_with_embeddings.parquet')

    df_code= df
    define_dataframe_types(df_code)

    code_embeddings = load_embeddings(df_code)
    df_code = df_code.set_index(idx_load)
    reset_messages_ai()

UPLOAD_DIR = "data_pdf"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.split('.')[0]
    sub_dir = UPLOAD_DIR+"/"+file_name
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    with open(os.path.join(sub_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(UPLOAD_DIR, uploaded_file.name)


def convert_text_token(file_name):
    pdf_folder = 'data_pdf'
    output_csv = 'data_sources/'+file_name+'.csv'
    process_folder(pdf_folder, output_csv)
    
    
def convert_token_embeddings(file_name):
    # Replace 'your_input.csv' with the path to your CSV file
    input_csv = 'data_sources/'+file_name+'.csv'
    process_dataframe(input_csv)

# Function to encode a file to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def create_download_link(title, filename):
    b64 = get_base64_of_bin_file(filename)
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{title}</a>'
    return href  

def read_text_from_file(faq_name):
    try:
        file_path = "messages_ai_debug.txt"
        with open(file_path, 'r') as file:
            # Read the entire contents of the file
            text_data = file.read()
            list_object = ast.literal_eval(text_data)
            l = []
            for d in list_object:
                l.append(d['content'])
            doc = Document()

    # Add each item from the list to the document as a paragraph
            for item in l:
                doc.add_paragraph(item)

                # Save the document
            doc.save(faq_name+".docx")
            #return text_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
if "temp" not in st.session_state:
    st.session_state["temp"] = ""

def clear_text():
    st.session_state["temp"] = st.session_state["user_input"]
    st.session_state["user_input"] = ""

def handle_userinput(user_question):
    
    #message = "What is the side effect of Olumiant?"
    chat_history = []
    user_question = user_question.lower()
    response, chat_history = respond(user_question, chat_history)   
              
    #response = st.session_state.conversation({'question': user_question})
    print('RESPONSE', response)
    st.session_state.chat_history = response
    for i, message in enumerate(response):
        a = 0
        
        if i % 2 == 0:
            print(i)
            print(message['role'])
            print(message['content'])
            st.write(user_template.replace(
                "{{MSG}}", message['content']), unsafe_allow_html=True)
            #st.session_state.chat_history.append({"role": message['role'], "content": message['content']})
            
            
        else:
            print(i)
            print(message['role'])
            print(message['content'])
            st.write(bot_template.replace(
                "{{MSG}}", message['content']), unsafe_allow_html=True)
            #st.session_state.chat_history.append({"role": message['role'], "content": message['content']})
            
    # Display download button
    
    faq_name = st.text_input("Please give the FAQ file name to download the file!!")
    if faq_name:
        read_text_from_file(faq_name)
        st.markdown(create_download_link("Click to download", faq_name+'.docx'), unsafe_allow_html=True)
    else:
        st.warning("Please give the FAQ file name to download the FAQ doc!!")
    feedback = streamlit_feedback(
    feedback_type="faces",
    optional_text_label="[Optional] Please provide an explanation",
    key="feedback"
    )        
        
        

        
            
    # for message in st.session_state.chat_history: # Display the prior chat messages
    #     with st.session_state.chat_history[0]:
    #         st.write(message["content"])
            
# def get_text_chunks(text):
#             text_splitter = CharacterTextSplitter(
#                 separator="\n",
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#             )
#             chunks = text_splitter.split_text(text)
#             return chunks
        

def respond(message, chat_history):
    
      
    global messages_ai
    global file_name_chat
    global chat_id
    messages_ai = st.session_state.chat_history
    print("Message_AI: ",messages_ai)

    bot_message, messages_ai = initiate_countdown(basic_chain_parallel_single, messages_ai, message,code_embeddings, df_code, seconds = 60)
    write_messages_ai_to_txt(messages_ai)
    print(messages_ai)
    chat_history.append((message, bot_message))  
    return messages_ai, chat_history  
    
def write_messages_ai_to_txt(messages_ai, file_path="messages_ai_debug.txt"):
    with open(file_path, mode='w', encoding='utf-8') as file:
        # Prepare a serializable version of messages_ai
        serializable_messages_ai = []
        for item in messages_ai:
            if isinstance(item, dict):
                # Ensure all values in the dict are serializable
                serializable_item = {k: (v if isinstance(v, (str, int, float, list, dict)) else str(v)) for k, v in item.items()}
                serializable_messages_ai.append(serializable_item)
            else:
                # Convert non-dict items to string
                serializable_messages_ai.append(str(item))
        
        try:
            messages_str = json.dumps(serializable_messages_ai, indent=4)
            file.write(messages_str)
        except TypeError as e:
            file.write(f"Error serializing messages_ai: {e}")

def main():
    load_dotenv()
    # if not st.session_state.response:
    #     print("Response: ", st.session_state.response)
    #     load_dfs("medical_data")
        #st.session_state.method_called = True
    st.set_page_config(page_title="Eli Lilly Med Affair",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        print("chat_history not in st.session_state")
        st.session_state.chat_history = []
        load_dfs("medical_data")

    st.header("Med affairs chat bot :books:")
    
    user_question = st.text_input("Ask a question about your documents:", key="user_input", on_change=clear_text)
    user_question = st.session_state.temp
    if user_question:
        handle_userinput(user_question)
        
    # if st.button("Clear"):
    #     st.markdown("<script>window.location.reload(true);</script>", unsafe_allow_html=True)
    #     st.session_state.chat_history = []
    #     load_dfs("medical_data")
        

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type=["pdf"])
        
        if pdf_docs is not None:
            # Save uploaded file
            file_path = save_uploaded_file(pdf_docs)
            st.success(f"PDF uploaded successfully: {file_path}")
        if st.button("Process"):
            if pdf_docs is not None:
                with st.spinner("Processing"):
                    # get pdf text
                    #raw_text = get_pdf_text(pdf_docs)
                    file_name = pdf_docs.name.split('.')[0]
                    file_path = os.path.join(UPLOAD_DIR,file_name)
                    convert_text_token(file_name)
                    convert_token_embeddings(file_name)
                    load_cusrrent_dfs(file_name)
                    st.success(f"{file_name} is processed successfully.")
                    
            else:
                st.warning("Please upload required document.")
                
        
    # get the text chunks
    #text_chunks = get_text_chunks(raw_text)
    
    

if __name__ == '__main__':
    # if "response" not in st.session_state:
    #     print("response not in st.session_state")
    #     st.session_state.method_called = False
    main()