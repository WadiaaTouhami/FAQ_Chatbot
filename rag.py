from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
from uuid import uuid4
from langchain_core.documents import Document
from typing import List, Dict
import os

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="FAQ_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)


def load_faq_to_chroma(faq_data: Dict, vector_store) -> None:
    """
    Load FAQ data into a Chroma vector store.

    Args:
        faq_data (Dict): Dictionary containing FAQ data with 'questions' key
        vector_store: Initialized Chroma vector store instance
    """
    # Convert FAQ entries to Documents
    documents = []
    for qa in faq_data["questions"]:
        # Combine question and answer in the page_content
        page_content = f"Question: {qa['question']}\nAnswer: {qa['answer']}"

        # Create Document with metadata to distinguish question/answer
        doc = Document(
            page_content=page_content,
            metadata={
                "type": "faq",
                "question": qa["question"],
                "answer": qa["answer"],
            },
        )
        documents.append(doc)

    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add documents to vector store
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Successfully loaded {len(documents)} FAQ entries into the vector store.")


def main():
    # Path to dataset
    json_path = "./Ecommerce_FAQ_Chatbot_dataset.json"

    # Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found at {json_path}")

    try:
        # Load JSON data
        with open(json_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        # Validate data structure
        if not isinstance(faq_data, dict) or "questions" not in faq_data:
            raise ValueError(
                "Invalid JSON format: Expected a dictionary with 'questions' key"
            )

        # Load the FAQ data into Chroma
        load_faq_to_chroma(faq_data, vector_store)

    except json.JSONDecodeError:
        print("Error: Invalid JSON file format")
        raise
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
