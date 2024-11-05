# langchain_app.py

# Plan:
# Ingest Project Files: Use LangChain to read and process the source code files for context.
# Split and Embed Documents: Split large files and embed them using OpenAI's embeddings API. Store these embeddings in a database for efficient retrieval.
# Query and Memory Management: Use embeddings to retrieve relevant parts of the code based on user queries. Maintain long-term memory by storing conversations in a database to manage context across requests.
# Generate Documentation: Use the GPT-4-turbo model to generate documentation based on specific parts of the code retrieved.


import os
import sqlite3
from langchain import OpenAI, DocumentLoader, VectorStore, ConversationChain
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings and model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
openai_model = OpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

# Database setup
db_connection = sqlite3.connect("langchain_memory.db")
db_cursor = db_connection.cursor()

# Create tables for storing context if not already present
db_cursor.execute('''CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, file_name TEXT, content TEXT, embedding BLOB)''')
db_cursor.execute('''CREATE TABLE IF NOT EXISTS conversation_history (id INTEGER PRIMARY KEY, input TEXT, response TEXT)''')
db_connection.commit()

def load_and_embed_files(directory_path):
    # Step 1: Load files from a directory
    loader = DirectoryLoader(directory_path, glob="**/*.py")  # Adjust the file pattern as needed
    documents = loader.load()

    # Step 2: Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Step 3: Embed and store documents in a vector database (Chroma)
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

def store_document_embeddings_in_db(vectorstore):
    # Store embeddings for persistence
    for doc_id, document in enumerate(vectorstore):
        db_cursor.execute(
            '''INSERT INTO documents (file_name, content, embedding) VALUES (?, ?, ?)''',
            (document.metadata.get("file_name", f"doc_{doc_id}"), document.page_content, document.embedding)
        )
    db_connection.commit()

def generate_documentation(query, vectorstore):
    # Step 1: Set up retrieval QA with vectorstore
    retrieval_qa = RetrievalQA(llm=openai_model, retriever=vectorstore.as_retriever())

    # Step 2: Generate documentation based on query
    response = retrieval_qa.run(query)

    # Store conversation history
    db_cursor.execute("INSERT INTO conversation_history (input, response) VALUES (?, ?)", (query, response))
    db_connection.commit()
    return response

def get_conversation_history():
    # Retrieve past conversations for context
    db_cursor.execute("SELECT input, response FROM conversation_history ORDER BY id DESC LIMIT 10")
    return db_cursor.fetchall()

def main():
    # Set up project directory for processing
    project_directory = "path/to/your/project"  # Adjust the path to your project directory
    vectorstore = load_and_embed_files(project_directory)

    # Persist embeddings in a database
    store_document_embeddings_in_db(vectorstore)

    # Query and generate documentation
    print("Enter your query (or type 'exit' to quit):")
    while True:
        user_query = input("> ")
        if user_query.lower() == "exit":
            break
        response = generate_documentation(user_query, vectorstore)
        print("Generated Documentation:\n", response)

    # Optionally, retrieve conversation history
    history = get_conversation_history()
    print("Recent Conversations:", history)

if __name__ == "__main__":
    main()



# Explanation
# File Loading and Splitting: The code uses DirectoryLoader to load .py files from a given directory, then splits them into manageable chunks with RecursiveCharacterTextSplitter.
# Embeddings and Vector Storage: Embeddings are created and stored in a Chroma vectorstore. Each document’s embedding is also persisted in SQLite, which helps if you need to reload them later without recalculating.
# RetrievalQA Chain: The RetrievalQA chain is used to handle user queries by retrieving relevant chunks of code and generating documentation.
# Memory Management: Conversation history is saved in SQLite to retain context over multiple sessions. A helper function retrieves the last 10 interactions, which can be useful for debugging or improving the model's contextual memory.
# Documentation Generation: Upon user input, the application retrieves and processes code snippets to produce documentation.
# Notes
# This setup saves embeddings and interactions to a database, enabling session persistence.
# Ensure that the environment variable for the OpenAI API key is correctly set in a .env file.
# Next Steps
# a. Integrate automatic tests for checking the accuracy of generated documentation.

# b. Extend the code to allow custom document splitting based on functions, classes, or other logical blocks.