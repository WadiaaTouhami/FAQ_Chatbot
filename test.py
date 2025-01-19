import google.generativeai as genai
from rag import vector_store
import os
from dotenv import load_dotenv

load_dotenv(".env")
Gemini_API_KEY = os.getenv("GEMINI_API_KEY")


def most_similar(question):
    results = vector_store.similarity_search_with_score(question, k=1)
    return results


question = "can I cancel an order"
context = most_similar(question)
'''
prompt = """You are a professional e-commerce customer service chatbot. Your role is to assist customers with their queries in a friendly and helpful manner.

Instructions:
1. Always start with a polite greeting
2. Address the specific user concern
3. Provide clear, actionable information
4. If the query is unclear or information is missing, politely ask for clarification
5. Maintain a professional tone throughout

User Question: {}

Relevant FAQ Information: {}

Please respond in a helpful and professional manner."""
'''

prompt = """You are an e-commerce chatbot,
Based on the given data please try to reply to user question,
if the information are irrelevant say that you didn't have any info about that.
Start always with greetings words like "Hello", "Thank you for reaching us!" ...
question: {}

Given data: {}"""

prompt = prompt.format(question, context)

# Configure Gemini API
genai.configure(api_key=Gemini_API_KEY)

# Create the generative model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

chat = model.start_chat(history=[])


def send_msg(msg):
    try:
        response = chat.send_message(msg)
        return response.text
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request. Please try again later. Error: {str(e)}"


print(send_msg(prompt))

print(chat.history)
