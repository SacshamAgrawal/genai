import os
from langchain import OpenAI
from langchain.vectorstores import FAISS
from langchain.document_loaders import WebBaseLoader  # Adjust if you're using a different loader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize OpenAI model
llm = OpenAI(model="gpt-4-turbo", temperature=0.5)

# Initialize vector store (e.g., FAISS)
vector_store = FAISS.load_local("path_to_your_vector_store")  # Load your vector DB

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Function to generate documentation for a specific code chunk
def generate_documentation(code_chunk):
    prompt = f"Please summarize the following code, detailing its purpose, inputs, outputs, and notable patterns:\n\n{code_chunk}"
    response = llm(prompt)
    return response

# Function to document each module or function
def document_code_module(module_name):
    # Retrieve relevant code chunks for the module from the vector store
    code_chunks = vector_store.similarity_search(module_name, k=5)  # Adjust k as needed
    
    documentation = []
    
    for code_chunk in code_chunks:
        # Generate documentation for each code chunk
        doc = generate_documentation(code_chunk.page_content)
        documentation.append(doc)
    
    # Combine all documentation for the module
    return "\n\n".join(documentation)

# Main documentation generation function
def generate_full_documentation(modules):
    full_documentation = {}
    
    for module in modules:
        print(f"Generating documentation for module: {module}")
        module_docs = document_code_module(module)
        full_documentation[module] = module_docs
    
    return full_documentation

# Function to format documentation for output
def format_documentation(documentation):
    formatted_output = "# Project Documentation\n\n"
    
    for module, doc in documentation.items():
        formatted_output += f"## Documentation for {module}\n\n{doc}\n\n---\n\n"
    
    return formatted_output


# List of modules to document
modules_to_document = ["module_name_1", "module_name_2"]  # Replace with actual module names

# Generate documentation
documentation = generate_full_documentation(modules_to_document)

formatted_documentation = format_documentation(documentation)

# Output the formatted documentation
output_file_path = "project_documentation.md"
with open(output_file_path, "w") as output_file:
    output_file.write(formatted_documentation)

print(f"Documentation has been generated and saved to {output_file_path}.")
