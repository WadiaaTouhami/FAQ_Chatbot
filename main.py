from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from rag import vector_store
import os
from dotenv import load_dotenv
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(".env")
Gemini_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=Gemini_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Chatbot API",
    description="A FastAPI implementation of an e-commerce chatbot using RAG and Gemini",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
chat = model.start_chat(history=[])


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    response: str
    context: List[Dict]


def get_similar_contexts(question: str):
    """Retrieve similar contexts from the vector store."""
    try:
        results = vector_store.similarity_search_with_score(question, k=2)
        # Convert Documents to dictionary format
        contexts = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
            for doc, score in results
        ]
        return contexts
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving contexts: {str(e)}"
        )


def generate_prompt(question: str, contexts: List[Dict]) -> str:
    """Generate the prompt for the LLM."""
    return """You are a professional e-commerce customer service chatbot. Your role is to assist customers with their queries in a friendly and helpful manner.

Instructions:
1. Always start with a polite greeting
2. Address the specific user concern
3. Provide clear, actionable information
4. If the query is unclear or information is missing, politely ask for clarification
5. Maintain a professional tone throughout

User Question: {question}

Relevant FAQ Information: {contexts}

Please respond in a helpful and professional manner.""".format(
        question=question, contexts=contexts
    )


@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {"message": "E-commerce Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that processes user questions and returns responses.

    Args:
        request (ChatRequest): The chat request containing the user's question

    Returns:
        ChatResponse: The chatbot's response and relevant contexts
    """
    try:
        # Get similar contexts
        contexts = get_similar_contexts(request.question)

        # Generate prompt
        prompt = generate_prompt(request.question, contexts)

        # Get response from LLM
        response = chat.send_message(prompt)

        return ChatResponse(response=response.text, context=contexts)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
