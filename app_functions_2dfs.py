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

#dummy entries 
documentation_content_dataset =""
documentation_embeddings_dataset =""
code_content_dataset = ""
code_embeddings_dataset = ""
secrets ="secrets.toml"

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#secrets
toml_dir = secrets

with open(toml_dir, 'r') as f:
    config = toml.load(f)

medicine_name = ""

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

#source code
df_code = pd.DataFrame()

#documentation
df_doc = pd.DataFrame()

#Encoding Variables
MAX_SECTION_LEN = 5000
SEPARATOR = " #### "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

#FUNCTION SECTION STARTS HERE
#Knowledge store embeddings and content
# Define the Azure OpenAI language generation function
def generate_text(message_hist, temp = 0.3, max_tok = 800):
    try:
        response = openai.ChatCompletion.create(
        engine=engine_name,
        #model=engine_name,
        messages=message_hist,
        temperature=temp,
        max_tokens=max_tok,
        stop=None          
    )
        logging.debug('generate_txt'+ response['choices'][0]['message']['content'])
        return response['choices'][0]['message']['content']
   
    except openai.error.Timeout:
      #Handle timeout error, e.g. retry or log
      return("OpenAI API request timed out.")
    
    except openai.error.APIError:
      #Handle API error, e.g. retry or log
      return("OpenAI API returned an API Error.")
    
    except openai.error.APIConnectionError:
      #Handle connection error, e.g. check network or log
      return("OpenAI API request failed to connect.")
  
    except openai.error.InvalidRequestError:
      #Handle invalid request error, e.g. validate parameters or log
      return("OpenAI API request was invalid.")
   
    except openai.error.AuthenticationError:
      #Handle authentication error, e.g. check credentials or log
      return("OpenAI API request was not authorized.")
    
    except openai.error.PermissionError:
      #Handle permission error, e.g. check scope or log
      return("OpenAI API request was not permitted.")
    
    except openai.error.RateLimitError:
      #Handle rate limit error, e.g. wait or log
      return("OpenAI API request exceeded rate limit.")
    
    except Exception:
      # Print the error message
      return("An Error occured. Please try again in a few moments.")


#Embeddings function
def get_embedding(text: str, engine: str=EMBEDDING_MODEL) -> list[float]:   
    try:
        result = openai.Embedding.create(engine=EMBEDDING_MODEL,input=text)
        emb = result["data"][0]["embedding"]
        return emb
    except openai.error.RateLimitError as e:
        time.sleep(20)
        result = openai.Embedding.create(engine=EMBEDDING_MODEL,input=text)
        emb = result["data"][0]["embedding"]
        return emb
    except Exception as e:
      # Print the error message
        print(f"EXCEPTION IN get_embeddings: {e}")
        return None

  #Compute embeddings function
def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:

    return {
        heading: get_embedding(r.content) for heading, r in df.iterrows()
    }

#Load embeddings function
def load_embeddings(df):

    #df = pd.DataFrame(dfa)
    #Initialize an empty dictionary to store the result
    result_dict = {}

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # Extract values from columns 'a', 'b', and 'c'
        a_value = row['title']
        b_value = row['heading']
        #d_value = row['idx']
        c_value = row['embeddings']

        # Create a tuple (a, b) as the key and assign the 'c' list as the value
        key = (str(a_value), str(b_value))
        result_dict[key] = c_value

    return result_dict

#Vector similarity function 
def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))


#Order document by similarity function
def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame, head, max_selection) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    i = 1

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        if isinstance(document_section['tokens'], (pd.DataFrame, pd.Series)):
            toknum = document_section['tokens'].values[0]
        else:
            toknum = document_section['tokens']

        if isinstance(document_section['content'], (pd.DataFrame, pd.Series)):
            document_sect = document_section['content'].values[0]
        else:
            document_sect = document_section['content']

        chosen_sections_len += toknum + separator_len
        if chosen_sections_len > max_selection:
            break
        context_heading = " Context {}-- Title: {}, heading: {}, Context Body:".format(i,section_index[0], section_index[1])
        i += 1
        chosen_sections.append(SEPARATOR + context_heading + document_sect.replace("\n", " ")) # add here str(section_index)
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    #print("\n".join(chosen_sections_indexes))
    header = head

    str_chosen_sections = str(chosen_sections)
    #print(str_chosen_sections) 
    return header + "".join(str_chosen_sections)

def construct_prompt2(question: str, context_embeddings: dict, df: pd.DataFrame, head, max_selection) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    i = 1

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        if isinstance(document_section['tokens'], (pd.DataFrame, pd.Series)):
            toknum = document_section['tokens'].values[0]
        else:
            toknum = document_section['tokens']

        if isinstance(document_section['content'], (pd.DataFrame, pd.Series)):
            document_sect = document_section['content'].values[0]
        else:
            document_sect = document_section['content']

        chosen_sections_len += toknum + separator_len
        if chosen_sections_len > max_selection:
            break
        context_heading = " Context {}-- Title: {}, heading: {}, Context Body:".format(i,section_index[0], section_index[1])
        i += 1
        chosen_sections.append(SEPARATOR + context_heading + document_sect.replace("\n", " ")) # add here str(section_index)
        chosen_sections_indexes.append(str(section_index))

    str_chosen_sections = str(chosen_sections)
    #print(str_chosen_sections) 
    return (str_chosen_sections)


#sleep for a random split of a second
def random_sleep():
    sleep_time = random.uniform(0.100, 0.999)
    time.sleep(sleep_time)

#PROMPTS
code_header = f"""You are an expert medical AI assistant with up-to-date knowledge for OLUMIANT, a medication developed by Eli Lilly. 
                OLUMIANT is a medication featuring the active ingredient baricitinib, primarily used to treat rheumatoid arthritis. 
                You excel in providing expert advice on this drug based on user inquiries. Your responses are rooted in integrity, excellence, 
                and respect for individuals, ensuring accuracy and reliability in all information shared. 
                Answer the user's query using only this Referenced information.\
                *** Reference information ***\
    
                """  
code_answering_instructions = """
[IMPORTANT INSTRUCTIONS]:
- If a query does not pertain to legitimate OLUMIANT related inquiry, respond with: "I'm sorry, I can only assist with OLUMIANT related queries."
- If the necessary information for answering "Q:" is not in the provided "Referenced information", say: "Sorry, my knowledge base lacks the relevant information to address your question."
- Upon finding the required information in the "Referenced information", address the query or complete the task ensuring you also follow the next point.
- Conclude all responses prior to providing reference source by either:
  1. Offering further assistance
  2. Providing options for further exploration
- Give detailed comprehensive answers, using all available useful information.
- Reference the source of your answer at the end, using the format: "Reference for answer: <Document Name>, <Page Number>."

\n Input Q:
"""

doc_embeddings = {}
code_embeddings = {}


def preprocess_user_input(user_input, message_history, temp=0.3, max_tok=300):
    """
    Enhances a user's query by incorporating context from the conversation history. 
    The function generates an optimized query that is more precise and informative, 
    using a model trained for dialogue review and query enhancement.

    """  
    # Building the dialogue history string from message_hist
    dialogue_history = ""
    for msg in message_history:
        if msg['role'] == 'user':
            dialogue_history += f"user_input:{msg['content']}\n"
        elif msg['role'] == 'assistant':
            dialogue_history += f"assistant_output:{msg['content']}\n"
        
    # Adding the last user input
    dialogue_history += f"Refine this query considering the conversation context, user_input:{user_input}"     
    user_message = f"""
        
    [DIALOGUE]:
    {dialogue_history}

    """  
    # System message explaining the task for internal model guidance
    system_message = """Enhance the user's query based on the conversation history. The conversation history consists of a user asking questions related to OLUMIANT, a drug containing Baricitinib as its active ingredient.
                     Your sole purpose is to pass employee queries to an Eli Lilly question answering expert who only has this question and no previous context.
                    Focus on refining and providing full context in the question. Only return the optimized query. Do not enchance question with previous reference information or enchance any with any information outside of chat history. 
                    Only refine and optimise when there is information in history to do so.
                    """
    
    # Constructing the input for the model
    messages =[{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}]

    # Generate the enhanced query using the predefined function
    try:
        enhanced_query = generate_text(messages, temp=temp, max_tok=max_tok)
        logging.debug("Original user query: %s", user_input)
        logging.debug("Enhanced user query: %s", enhanced_query)
        #if enhanced_query == "OpenAI API request was invalid.":
        #    return user_input
        #else:
        #    return enhanced_query
        return enhanced_query
    except Exception as e:
        logging.error("Error in query enhancement: %s", str(e))
        # Fallback to the original input in case of an error
        return user_input

def ask_gpt(user_input, embeddings, df, header, instructions, message_hist, max_selection=2500, max_tokens_output=800, temperature=0.3):
    logging.debug("Before preprocessing: Length of message_hist: %d", len(message_hist))
    logging.debug("Contents of message_hist: %s", message_hist)
    
    if len(message_hist) >= 2:
        logging.debug("preprocess_user_input being called")
        user_input = preprocess_user_input(user_input, message_hist, temp=temperature, max_tok=max_tokens_output)
        logging.debug("Updated user input based on conversation history: %s", user_input)
    else:
        logging.debug("preprocess_user_input NOT called due to message_hist length <= 2")
   
    random_sleep()
    # Extract the last 6 rows from df_code to create a new dataframe
    
    new_df = df.iloc[-6:].copy()
    df = df.iloc[:-6]

    # Since embeddings might be a dictionary, handle it appropriately
    if isinstance(embeddings, dict):
        # Convert dictionary to DataFrame for easier manipulation
        index = pd.MultiIndex.from_tuples(embeddings.keys(), names=['Document Title', 'Section Title'])
        code_embeddings_df = pd.DataFrame(list(embeddings.values()), index=index)
        
        # Splitting code_embeddings DataFrame
        new_code_embeddings = code_embeddings_df.iloc[-6:].copy()
        code_embeddings_df = code_embeddings_df.iloc[:-6]

        # Optionally convert back to dictionary if necessary
        embeddings = {idx: row.tolist() for idx, row in code_embeddings_df.iterrows()}
        new_code_embeddings_dict = {idx: row.tolist() for idx, row in new_code_embeddings.iterrows()}

    elif isinstance(embeddings, pd.DataFrame):
        # If code_embeddings is already a DataFrame, simply split it
        new_code_embeddings = embeddings.iloc[-6:].copy()
        embeddings = embeddings.iloc[:-6]

    sys_prompt = construct_prompt(user_input, new_code_embeddings_dict, new_df, header,4000)
    
    sys_prompt = sys_prompt + construct_prompt2(user_input, embeddings, df, header,2000)

    print(sys_prompt)

    # Initialize the messages list with the system context
    messages = [{'role': 'system', 'content': sys_prompt}]
    
    
    # Add the (potentially updated) user message
    messages.append({'role': 'user', 'content': instructions + user_input})
    
    # Logging the final messages before generating the response
    logging.debug("Final messages before generating response: %s", messages)
    
    # Generate the response using the updated user input and system context
    response = generate_text(messages, temp=temperature, max_tok=max_tokens_output)
    
    # Logging the generated response
    logging.debug("Generated response: %s", response)
    
    return response


def initiate_countdown(my_function, *args, seconds=5, **kwargs):
    result = None
    prompt = None

    def execute_function():
        nonlocal result, prompt
        try:
            result, prompt = my_function(*args, **kwargs)
        except Exception as e:
            result, prompt = "EXCEPTION OCCURED AFTER TIMEOUT:"+str(e), [{"role": "system", "content": "You are a helpful AI GPT expert."}]

    timer_thread = threading.Thread(target=execute_function)
    timer_thread.start()
    timer_thread.join(seconds)

    if timer_thread.is_alive():
        result = "The Server is very busy and can't handle your request at the minute, please try again in a few moments."
        prompt = [{"role": "system", "content": "You are a helpful AI GPT expert."}]
    return result, prompt


def run_in_parallel(*functions_with_args):
    # Define a function to be executed by each thread
    def run_function(func, args, result_list, index):
        try:
            result_list[index] = func(*args)
        except Exception as e:
            result_list[index] = e

    num_functions = len(functions_with_args)
    results = [None] * num_functions
    threads = []

    # Create and start a thread for each function with its arguments
    for i, (func, args) in enumerate(functions_with_args):
        thread = threading.Thread(target=run_function, args=(func, args, results, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return results

def input_checker2(message_hist, user_input, temp=0):
    random_sleep()
    if len(message_hist) < 2:
        logging.debug('input_checker_called less than 2')
        return "c"
    else:
        logging.debug('input_checker_called more than 2')

        # Building the dialogue history string from message_hist
        dialogue_history = ""
        for msg in message_hist:
            if msg['role'] == 'user':
                dialogue_history += f"user_input:{msg['content']}\n"
            elif msg['role'] == 'assistant':
                dialogue_history += f"assistant_output:{msg['content']}\n"
        
        # Adding the last user input
        dialogue_history += f"last_user_input:{user_input}"

        system_message = """You are a dialogue evaluator. 
        Your task is to evaluate the dialogue history between an employee and a HR employee assistant."""
        
        user_message = f"""[INSTRUCTIONS]:
        You will output only one letter that best satisfies one of the following conditions:
        -With a severity level set to very low, if the "last_user_input" is harmful or offensive, output: "a"
        -If the "last_user_input" is a continuation of the exact same idea of the previous inputs and outputs, output letter: "b"
        -If the "last_user_input" is a new query or a change of subject or idea, output letter: "c"
        -If the "last_user_input" is out not not a valid HR related query, output letter: "d"
        
        [DIALOGUE]:
        {dialogue_history}

        Output a single letter and nothing else

        Example output: 'b'
        """

        prompt = [{"role": "system", "content": system_message},
                  {"role": "user", "content": user_message}]
                  
        #return generate_text(prompt, temp).lower().replace(" ", "")
        return "c"

    #else:
    #    return "c"


def basic_chain_parallel_single(message_hist, user_input,code_embeddings, df_code):

    user_input = str(user_input)

    answers = run_in_parallel(
            (input_checker2, (message_hist, user_input, 0,)),         
            (ask_gpt, (user_input, code_embeddings, df_code, code_header, code_answering_instructions,message_hist, 5000, 800, 0.2,)),
            )
    checker = answers[0]

    # Log the output of input_checker2
    logging.debug(f"Checker result: {checker}")

    if checker == "a":
        last_response = "Your input may not be appropriate in terms of content moderation. Please refrain from making such comments and try something else."
        #print("\n***INTENT_CHECKER=", checker, "\n")
        return last_response , message_hist
    elif checker == "d":
        last_response = "I'm sorry, I can only assist with OLUMIANT related queries."
        #print("\n***INTENT_CHECKER=", checker, "\n")
        return last_response, message_hist   
    
    else:
        last_response = answers[1]
        
        # message_hist.append({"role": "user", "content": user_input})
        # message_hist.append({"role": "assistant", "content": last_response})
        message_hist.insert(0, {"role": "user", "content": user_input})
        message_hist.insert(1, {"role": "assistant", "content": last_response})
        print("MESSAGE_HIST: ", message_hist)

        #print("\n@@@ FINAL CHAT @@@\n",last_prompt,"\n")
        return last_response,message_hist# last_prompt