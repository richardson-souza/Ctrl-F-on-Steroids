from data_loader import DataLoader
from vector_store import VectorStoreManager
import config


def main():
    """
    Main function to orchestrate the document indexing process.

    This script performs the following steps:
    1. Initializes a `DataLoader` with the repository path specified in `config.REPO_PATH`.
    2. Calls the `load_repository_documents` method of the `DataLoader` to scan
       the repository, process allowed file types, and generate a list of
       Langchain `Document` objects.
    3. If documents are found, it initializes a `VectorStoreManager`.
    4. Calls the `create_store_from_documents` method of the `VectorStoreManager`
       to create (or overwrite) a Chroma vector store using the loaded documents
       and the embedding model specified in the configuration.
    5. Prints status messages throughout the process and a completion message.
    """
    print("Starting the indexing process...")

    loader = DataLoader(repo_path=config.REPO_PATH)

    documents = loader.load_repository_documents()

    if not documents:
        print(
            "No documents were found to index. Please check your REPO_PATH and configurations."
        )
        return

    vector_store_manager = VectorStoreManager()

    vector_store_manager.create_store_from_documents(documents=documents, is_test=config.TEST_COLLECTION)

    print("✅ Indexing complete.")


if __name__ == "__main__":
    main()
