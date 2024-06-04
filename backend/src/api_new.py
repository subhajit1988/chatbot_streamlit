import pandas as pd
import tiktoken
import openai
import numpy as np
import toml
#from preprocess import *
#from embeddings import *
#from lmc_function import *
from flask import Flask, render_template, request, jsonify, session
import re
from flask_swagger_ui import get_swaggerui_blueprint
#from app_function_doc import *
from app_functions_2dfs import *
from flask_cors import CORS

secrets ="../../secrets.toml"

#secrets
toml_dir = secrets

with open(toml_dir, 'r') as f:
    config = toml.load(f)

# Set up OpenAI API credentials
client = AzureOpenAI(
    azure_endpoint=config['api_base'],
    api_key=config['api_key'],
    api_version=config['api_version'],
)


# # Set up OpenAI API credentials for embedding large
# embclient = AzureOpenAI(
#     azure_endpoint="https://openai-gbs-openai-ritm4758276-instance.openai.azure.com/",
#     api_key="25733a225d5e463b80a33ee75ad9e9e4",
#     api_version="2024-03-01-preview",
# )
# Set up engine name
engine_name = config['engine_name']


#Completion on knowlede store config
EMBEDDING_MODEL = config['EMBEDDING_MODEL']
EMBEDDING_MODEL = "bot-text-embedding-3-large"

def define_dataframe_types(df):
    df.loc[:, 'heading'] = df['heading'].astype(str)
    df.loc[:, 'title'] = df['title'].astype(str)
    df.loc[:, 'tokens'] = df['tokens'].astype(int)
    df.loc[:, 'content'] = df['content'].astype(str)
    
    
def reset_messages_ai():
    global messages_ai
    # Reset messages_ai to its default state or an empty list/dictionary, depending on your implementation
    messages_ai = []
    

def load_dfs(local_toml, idx_load = ['title','heading']):
    global data_sources
    global doc_embeddings 
    global code_embeddings 
    global df_code
    global df_doc
    global app_prompt_list
 
    df = pd.read_parquet(config[local_toml])
    df = pd.read_parquet("../../data_sources/EU_cFAQ_ABE022_BIOMARKERS_with_embeddings.parquet")

    df_code= df
    define_dataframe_types(df_code)

    code_embeddings = load_embeddings(df_code)
    df_code = df_code.set_index(idx_load)
    reset_messages_ai()
    
def convert_text_token():
    pdf_folder = 'data_doc'
    output_csv = 'data_sources/EU_cFAQ_ABE022_BIOMARKERS.csv'

    process_folder(pdf_folder, output_csv)
    
def convert_token_embeddings():
    # Replace 'your_input.csv' with the path to your CSV file
    input_csv = 'data_sources/EU_cFAQ_ABE022_BIOMARKERS.csv'
    process_dataframe(input_csv)
    
    
def respond(message, chat_history):
    global messages_ai
    global file_name_chat
    global chat_id
    print("Message_AI: ", messages_ai)

    # Process the incoming message and generate the bot's response
    #bot_message, messages_ai = basic_chain_parallel_single(messages_ai, message, code_embeddings, df_code)
    bot_message, messages_ai = initiate_countdown(basic_chain_parallel_single, messages_ai, message,code_embeddings, df_code, seconds = 60)


    # # Use regular expressions to find URLs and convert them to HTML anchor tags
    # bot_message = re.sub(
    #     r'(https?://\S+)',  # Regex to find URLs that start with http or https
    #     r'<a href="\1" target="_blank">\1</a>',  # Replacement pattern to create a clickable link
    #     bot_message  # The text in which URLs are to be replaced
    # )

    # Add the original and transformed messages to the chat history
    chat_history.append((message, bot_message))
    
    # Return the updated AI messages and chat history
    return messages_ai, chat_history


# Create a Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'chat_history' 
# Swagger configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "app",  # Replace with your app name
        'validatorUrl': None  # Disable Swagger validator
    }
)
app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix=SWAGGER_URL)


@app.route('/')
def initialize():
    #convert_parque()  
    #convert_text_token()
    #convert_token_embeddings()
    load_dfs("medical_data")
    return "The session is initialized"

import traceback
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_question = request.form.get('question')
        global messages_ai
        global file_name_chat
        global chat_id
        #print("Message_AI: ", messages_ai)
        chat_history = []
        response, chat_history = respond(user_question, chat_history)
        headers = {
            "Access-Control-Allow-Origin": "*", 
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "*"
        }
        return jsonify({'response': response}), 200, headers
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        error_details = traceback.format_exc()
        print(error_message)
        print(error_details)
        headers = {
            "Access-Control-Allow-Origin": "*", 
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "*"
        }
        return jsonify({'error': error_message, 'details': error_details}), 500, headers



if __name__ == '__main__':
    app.run(debug=False)