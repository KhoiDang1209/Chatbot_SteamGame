# 🎮 Steam Game Recommendation Chatbot

An intelligent chatbot that recommends Steam games using Retrieval-Augmented Generation (RAG). It integrates Google Gemini Flash API for smart responses and MongoDB for fast vector search. Built with MongoDB Vector Database, Streamlit, and Google Generative AI

## ✨ Features

- 🔍 **Game Search with RAG**: Combines Gemini Flash API and MongoDB vector search for accurate game recommendations.
- 🧠 **Context-Aware Chatbot**: Understands your queries about game genres, ratings, release years, and more.
- 🖥️ **Streamlit UI**: Simple and modern user interface to interact with the chatbot in real time.
- 💾 **MongoDB Vector Store**: Embedding-powered game data storage using Sentence Transformers for efficient similarity search.

# Demo: 
![image](https://github.com/user-attachments/assets/e6fa965e-23b2-4afa-8db3-b9b2fd9b00b2)

## 🚀 Getting Started

To get started with this Streamlit chatbot application, follow the steps below:

### 1. Installation

1.1 Clone this repository to your local machine:

```bash
git clone https://github.com/KhoiDang1209/Chatbox_SteamGame.git
```
1.2 Create a virtual environment:
```bash
python -m venv venv
```

1.3 Install required libraries:

```bash
pip install -r requirements.txt
```
1.4 Run preprocessing_game.ipynb to process data and import to MongoDB. Replace with your MongoDB access url.

### 2. Running Pipeline

2.1 Run the RAG_Steam_Game_Recommendation_Vector_Search.ipynb to embed game names and game descriptions. Then import embedded data to a new collection.

2.2 Run the RAG_Steam_Game_Recommendation_run_pipeline.ipynb to get the pipeline of the RAG system.

2.3 Run the Streamlit application with the following command:
```bash
streamlit run app.py
```

### 3. Other Features

 🚀 Replace other embedding model with specific task

 🚀 Prompt inference the model for specified instruction.

### By following these steps, you should be able to set up and run the Streamlit applications smoothly on your local machine.
