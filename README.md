# E-commerce FAQ Chatbot

A smart chatbot system designed for e-commerce platforms that leverages Retrieval-Augmented Generation (RAG) to provide accurate and contextual responses to customer queries. The system combines a comprehensive FAQ database with Google's Gemini AI to deliver professional and friendly customer service.

## Demo

[Watch the demo video](https://drive.google.com/file/d/1Luuzoqv2oeh9iGLXlfcCzsXpUWGdnR7o/view?usp=drive_link)

## Features

- RAG-based information retrieval using Chroma vector store
- Integration with Google's Gemini 1.5 Flash model
- FastAPI backend with automatic API documentation
- Containerized deployment with Docker
- Simple web interface for user interaction
- Efficient similarity search for relevant FAQ matching

## Dataset Description

The project uses an e-commerce FAQ dataset that includes:

- Common customer account management queries
- Payment and transaction-related questions
- Shipping and delivery information
- Product-related inquiries
- Return and refund policies

Dataset available at: [E-commerce FAQ Chatbot Dataset](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset/data)

## Project Structure

```
/
├── chroma_langchain_db/          # Vector database storage
├── Ecommerce_FAQ_Chatbot_dataset.json  # Source FAQ dataset
├── index.html                    # Web interface for the chatbot
├── main.py                       # FastAPI backend implementation
├── rag.py                        # Vector database creation and management
├── requirements.txt              # Python package dependencies
├── .env                          # Environment variables configuration
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── docker-compose.yml            # Docker services configuration
└── Dockerfile                    # Container build instructions
```

## System Requirements

- Python 3.10 or higher
- Docker and Docker Compose (for containerized deployment)
- 8GB RAM (minimum)
- 20GB free disk space

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/WadiaaTouhami/FAQ_Chatbot.git
cd FAQ_Chatbot
```

2. Set up environment variables:

```bash
cp .env.example .env
```

- Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Update the `.env` file with your API key:

```
GEMINI_API_KEY="your-api-key-here"
```

3. Download the [E-commerce FAQ dataset](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset/data) and place it in the project root.

## Installation

### Method 1: Local Development

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
.\venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize the vector database:

```bash
python rag.py
```

4. Start the application:

```bash
python main.py
```

### Method 2: Docker Deployment

1. Ensure Docker Engine is running

2. Build and start the containers:

```bash
docker-compose up -d --build
```

## Usage

Access the chatbot interface at: `http://localhost:8000`

The API documentation is available at: `http://localhost:8000/docs`

## Dependencies

Key packages used in this project:

- langchain-chroma==0.1.2
- langchain-huggingface==0.0.3
- google-generativeai==0.8.3
- fastapi==0.109.1
- uvicorn==0.27.0
- python-dotenv==1.0.0
- pydantic==2.6.1

## License

[MIT License](LICENSE)
