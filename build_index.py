import streamlit as st
import pandas as pd
import torch
import json
from glob import glob
from pathlib import Path
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core.service_context import set_global_service_context
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import download_loader

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Document
import bitsandbytes
from llama_index.core.node_parser import SentenceSplitter

# Define your system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information. <</SYS>>
"""  # Llama2's official system prompt

def model_tokenizer_embedder(model_name, auth_token):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="./model/",
        use_auth_token=auth_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="./model/",
        use_auth_token=auth_token,
        torch_dtype=torch.float16,
        load_in_8bit=False,  # Disabled 8-bit quantization for compatibility
    )

    embedding_llm = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    return tokenizer, model, embedding_llm

def load_metadata(file_path, file_type='json'):
    documents = []
    try:
        if file_type == 'json':
            with open(file_path, 'r') as f:
                metadata_list = json.load(f)
                for metadata in metadata_list:
                    # Combine title and abstract into a single text
                    text = f"Title: {metadata['title']}\nAbstract: {metadata['abstract']}"
                    doc = Document(text=text, metadata=metadata)
                    documents.append(doc)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                authors = row['authors'].split(';') if isinstance(row['authors'], str) else row['authors']
                categories = row['categories'].split(';') if isinstance(row['categories'], str) else row['categories']
                text = f"Title: {row['title']}\nAbstract: {row['abstract']}"
                metadata = {
                    "id": row['arxiv_id'],
                    "authors": authors,
                    "categories": categories
                }
                doc = Document(text=text)
                documents.append(doc)
        else:
            raise ValueError("Unsupported file type. Use 'json' or 'csv'.")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        raise
    return documents

def build_and_save_index(documents, index_path="vector_index"):
    # Define Settings for the index
    # settings = Settings(
    #     llm=None,  # Not needed for indexing
    #     embed_model=LangchainEmbedding(
    #         HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #     ),
    #     node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=20),
    #     chunk_size=512,
    #     num_output=512,
    #     context_window=4096,
    # )
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 4096
    
    # Create the vector store index
    index = VectorStoreIndex.from_documents(documents, embed_model = Settings.embed_model)
    
    # Persist index to disk
    index.storage_context.persist(index_path)
    print(f"Index successfully saved to {index_path}")

def main():
    # Initialize model components
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    auth_token = "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"  # Replace with your actual Hugging Face token
    
    print("Loading model and tokenizer...")
    tokenizer, model, embedding_llm = model_tokenizer_embedder(model_name, auth_token)
    
    # Path to your metadata file
    metadata_file = "/Users/varunravi/Desktop/Scientific Literature Explorer/filtered_arxiv_papers_metadata.csv"
    
    print(f"Loading metadata from {metadata_file}...")
    documents = load_metadata(metadata_file, file_type='csv')
    
    print("Building the vector store index...")
    build_and_save_index(documents, index_path="vector_index.json")

if __name__ == "__main__":
    main()