import openai
import pandas as pd
import os
import toml
import io

secrets ="secrets.toml"
#secrets
toml_dir = secrets

with open(toml_dir, 'r') as f:
    config = toml.load(f)

# Set up OpenAI API credentials
# azure required env vars commented out 
#openai.api_type = config['api_type']
#openai.api_base = config['api_base']
#openai.api_version = config['api_version']
openai.api_key = config['api_key']
# Set up engine name
engine_name = config['engine_name']

#Completion on knowlede store config
EMBEDDING_MODEL = config['EMBEDDING_MODEL']

def get_embeddings(text):
    """Get embeddings for a given text using OpenAI's Ada 2 model."""
    response = openai.Embedding.create(
        input=[text],
        engine="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def process_dataframe(input_csv):
    
    df = pd.read_csv(input_csv)
    #df = df[df['title']=='leaflet']

    # Apply embeddings to each content cell and store the list of embeddings
    df['embeddings'] = df['content'].apply(get_embeddings)
    print(df['title'])
    # Since each item in 'embeddings' is a list, direct saving to CSV will serialize this list as a string representation
    output_csv = input_csv.replace('.csv', '_with_embeddings.csv')
    df.to_csv(output_csv, index=False)
    
    # Saving to a Parquet file preserves the list structure in 'embeddings' more naturally
    output_parquet = input_csv.replace('.csv', '_with_embeddings.parquet')
    df.to_parquet(output_parquet, index=False)

    print(f"Output saved as {output_csv} and {output_parquet}")


