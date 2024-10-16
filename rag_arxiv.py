import pandas as pd
import torch
import json
from glob import glob
from pathlib import Path
from llama_index.core.prompts.prompts import SimpleInputPrompt
# Removed Streamlit imports
# from llama_index.core.service_context import set_global_service_context
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import download_loader

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Document
import bitsandbytes
from llama_index.core.node_parser import SentenceSplitter
import os
from llama_index.core import StorageContext, load_index_from_storage
import sys  # Added for error handling

# Define your system prompt
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information. <</SYS>>
"""  # Llama2's official system prompt


def model_tokenizer_embedder(model_name, auth_token):
    """
    Loads the tokenizer and model from Hugging Face and sets up the embedding model.
    
    Args:
        model_name (str): The name of the Hugging Face model to load.
        auth_token (str): Hugging Face authentication token.
        
    Returns:
        tuple: (tokenizer, model, embedding_llm)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="./model/", use_auth_token=auth_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="./model/",
            token=auth_token,
            torch_dtype=torch.float32,
            load_in_8bit=False,  # Disabled 8-bit quantization for compatibility
        )
    
        embedding_llm = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
    
        return tokenizer, model, embedding_llm
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}", file=sys.stderr)
        sys.exit(1)  # Replaces st.error and st.stop()


def load_index(index_path="vector_index.json"):
    """
    Loads the pre-built vector index from disk.
    
    Args:
        index_path (str): Path to the saved vector index.
        
    Returns:
        VectorStoreIndex: The loaded vector store index.
    """
    if not os.path.exists(index_path):
        print(f"Index file '{index_path}' not found. Please build the index first.", file=sys.stderr)
        sys.exit(1)  # Replaces st.error and st.stop()
    try:
        index = VectorStoreIndex.load_from_disk(index_path)
        return index
    except Exception as e:
        print(f"Error loading vector index: {e}", file=sys.stderr)
        sys.exit(1)  # Replaces st.error and st.stop()


def main():
    # Initialize model components
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    auth_token = "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"  # Replace with your actual Hugging Face token

    print("Loading model and tokenizer...")
    tokenizer, model, embedding_llm = model_tokenizer_embedder(model_name, auth_token)

    # Define the query wrapper prompt
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    # Initialize the LLM
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        model=model,
        tokenizer=tokenizer,
    )

    # Define Settings
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 4096

    # Load the index
    print("Loading vector index...")
    # Rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="vector_index.json")

    # Load index from the storage context
    new_index = load_index_from_storage(storage_context)
    query_engine = new_index.as_query_engine()

    # Get user input from the console
    prompt = input("Enter your prompt: ")
    if prompt:
        print("Generating response...")
        try:
            response = query_engine.query(prompt)
            print(response.response)
        except Exception as e:
            print(f"Error during query execution: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()