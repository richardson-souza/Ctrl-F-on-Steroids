# --- Centralized Configuration ---

# ❗ IMPORTANT: Change this path to the absolute path of your code repository.
# Example on macOS/Linux: "/Users/yourname/projects/my-airflow-dags"
# Example on Windows: "C:\\Users\\yourname\\projects\\my-airflow-dags"
REPO_PATH = "./data/para/seu/repositorio"

# --- Model Configuration ---
# The model used to create vector embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# The local LLM used to answer questions
LLM_MODEL = "gemma2:2b" # "codellama:7b-instruct"

# --- Vector Store Configuration ---
# The directory to store the persistent vector database
VECTOR_DB_PATH = "./chroma_db"

# --- Data Loader Configuration ---
# File extensions to include in the search
ALLOWED_EXTENSIONS = {".py", ".sql", ".md", ".json", ".yaml"}

