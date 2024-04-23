import fitz  # PyMuPDF
import csv
import os
import re
import tiktoken
import unicodedata
import re

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def clean_text(text: str) -> str:
    """Cleans the extracted text by fixing encoding issues and replacing common misinterpretations."""
    # Normalize unicode characters to their closest equivalent representation
    text = unicodedata.normalize('NFKD', text)
    
    # Dictionary of common misinterpretations and their corrections
    replacements = {
        'â€™': "'",    # Apostrophe
        'â€œ': '"',    # Opening double quote
        'â€': '"',    # Closing double quote
        'â€¢': '•',    # Bullet point
        'â€”': '—',    # Em dash
        'â€“': '–',    # En dash
        'â€˜': "'",    # Opening single quote
        'â€™': "'",    # Closing single quote
        'â€¦': '…',    # Ellipsis
        # Add more replacements as needed
    }
    
    # Apply replacements
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Replace newlines and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text


def extract_and_clean_text_by_pages(pdf_path):
    doc = fitz.open(pdf_path)
    title = os.path.basename(pdf_path).replace('.pdf', '')
    data = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()  # Extract all text from the page
        if text:  # Only add non-empty pages
            cleaned_text = clean_text(text)  # Clean the text
            data.append({
                'title': title,
                'heading': f'Page {page_num}',
                'content': cleaned_text,
                'tokens': num_tokens_from_string(cleaned_text)
            })

    return data

def write_to_csv(data, output_csv):
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csv_file:  # Change to append mode
        fieldnames = ['title', 'heading', 'content', 'tokens']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        for row in data:
            writer.writerow(row)

def process_folder(pdf_folder, output_csv):
    # Ensure the output CSV file has the headers before appending data
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['title', 'heading', 'content', 'tokens']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    for root, dirs, files in os.walk(pdf_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                data = extract_and_clean_text_by_pages(pdf_path)
                write_to_csv(data, output_csv)


