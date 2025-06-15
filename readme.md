# CTRL-F-ON-STEROIDS: A Conversational Interface for Your Repository

This project provides a powerful, locally-run search engine that allows you to ask natural language questions about your codebase. It uses a Retrieval-Augmented Generation (RAG) pipeline to understand the structure and content of your files, enabling you to quickly find information about DAGs, table schemas, SQL queries, and Python logic without ever sending your code to an external service.

🎯 Project Goal and Audience
Note: This project was developed as a learning experience and is intended to be used for educational purposes. It serves as a comprehensive template and a step-by-step guide for anyone looking to build their own local, AI-powered code search tools.

✨ Features
100% Local & Private: Your code is never sent over the internet. The entire process, from indexing to question-answering, runs on your local machine using Ollama.

Intelligent Pre-processing: The tool doesn't just read files as plain text. It has specialized parsers to understand the structure of:

JSON data dictionaries (extracting table and column descriptions).

YAML DAG configurations (extracting owners, schedules, tasks, and execution flow).

SQL files (extracting source tables and handling non-standard commands).

Python scripts (extracting imports and function definitions with docstrings).

Hybrid Search: For each file, we index both the full, raw content (for broad, keyword-like searches) and the smart, pre-processed chunks (for precise, semantic answers).

Advanced Self-Querying: The tool uses the LLM itself to analyze your question and automatically build a filtered query, ensuring it only searches through the most relevant documents. This prevents "context contamination" and dramatically improves accuracy.

⚙️ How It Works: The RAG Pipeline
Indexing (indexer.py): The script reads all files in your specified repository. It uses the DataLoader to parse each file type, creating intelligent summaries and extracting key metadata. An embedding model (all-MiniLM-L6-v2) then converts these text chunks into numerical vectors, which are stored in a local ChromaDB vector database.

Retrieval (ask.py): When you ask a question, the SelfQueryRetriever uses a local LLM (gemma2:2b) to understand your intent and create a filtered search query based on the metadata (e.g., filtering by dag_id or table name).

Generation (ask.py): The retriever fetches the most relevant document chunks from the database. These chunks, along with your original question, are passed to the LLM, which generates a final, context-aware answer.

🚀 Setup and Installation
Follow these steps to get the tool up and running.

- Prerequisites
Python 3.8+

Ollama installed and running on your machine.

- Clone the Repository
Clone this project to your local machine.

git clone [https://github.com/richardson-souza/Ctrl-F-on-Steroids.git](https://github.com/richardson-souza/Ctrl-F-on-Steroids.git)
cd Ctrl-F-on-Steroids

- Install Dependencies
Create a virtual environment and install the required Python packages.

## Create and activate virtual environment

python3 -m venv venv
source venv/bin/activate

## Install requirements

pip install -r requirements.txt

The requirements.txt file should contain:

langchain
langchain_community
langchain-huggingface
langchain-chroma
langchain-ollama
sentence-transformers
chromadb
PyYAML
sql-metadata

- Download the Local LLM
Pull the language model that will be used for answering questions.

ollama pull gemma2:2b

- Configure Your Repository Path
This is a mandatory step. Open the config.py file and update the REPO_PATH variable to point to the absolute path of the code repository you want to search.

🛠️ Usage
Using the tool is a two-step process.

Step 1: Index Your Repository
Run this command once to process your code and build the local vector database. You only need to re-run this when your code has changed significantly.

python indexer.py

This will create a chroma_db directory in your project folder.

Step 2: Ask Questions
Start the interactive Q&A tool by running:

python ask.py

Once the tool is ready, you can start asking questions.

Example Questions:

Who is the owner of the DAG 'X'?

What is the purpose of the Y function?

Which SQL scripts read from the table 'XYZ'?

Which DAGs run on the schedule '0 5 * * 1'?

📂 Project File Structure
config.py: Central configuration for file paths, model names, and allowed extensions.

data_loader.py: The core of the pre-processing logic. Contains specialized classes (JsonProcessor, YamlProcessor, etc.) to parse different file types.

vector_store.py: Manages the creation and loading of the ChromaDB vector store.

indexer.py: Standalone script to run the indexing process.

ask.py: The main application file that runs the Q&A loop and contains the SelfQueryRetriever.

test_*.py: Unit and integration tests to ensure the components work as expected.

🔧 Customization
⚠️ Important Note on Adaptation

The intelligent pre-processing logic in data_loader.py is the heart of this tool's effectiveness. The JsonProcessor, YamlProcessor, SqlProcessor, and PythonProcessor classes have been specifically tailored to understand the unique structure of the files in this project's example repository.

When you use this tool on your own codebase, you will almost certainly need to adapt the logic inside these processor classes. Think of them as a template. You should inspect your own file structures and modify the process method in each class to correctly parse your data and extract the relevant metadata. This customization is what makes the tool so powerful.
