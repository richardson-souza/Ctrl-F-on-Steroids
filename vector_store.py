import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.docstore.document import Document

import config


class VectorStoreManager:
    """
    Manages the creation, persistence, and retrieval of a Chroma vector store.

    This class encapsulates the logic for initializing the embedding model,
    setting up a persistent ChromaDB client, creating a new vector store
    from a list of documents, and loading an existing vector store.

    Attributes:
        embedding_function (HuggingFaceEmbeddings): The embedding model instance
            used to convert text documents into vector embeddings.
        persist_directory (str): The file system path where the ChromaDB
            vector store is persisted.
        client (chromadb.PersistentClient): The ChromaDB client instance for
            managing the persistent database.
    """

    def __init__(self):
        """
        Initializes the VectorStoreManager.

        Sets up the HuggingFace embedding model specified in the configuration
        (forcing CPU usage) and initializes a persistent ChromaDB client using
        the path from the configuration.
        """
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # Use CPU for embeddings
        )
        self.persist_directory = config.VECTOR_DB_PATH
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def create_store_from_documents(self, documents: List[Document]):
        """
        Creates a new Chroma vector store from a list of documents and persists it.

        This method takes a list of Langchain `Document` objects, embeds them
        using the initialized embedding function, and stores them in a new
        Chroma database at the configured `persist_directory`.

        Args:
            documents: A list of `langchain.docstore.document.Document` objects
                       to be added to the vector store.
        """
        print("Creating new vector store...")
        Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        print("Vector store created and saved.")

    def get_vector_store(self) -> Chroma:
        """
        Loads and returns an existing Chroma vector store from the persist directory.

        Checks if the `persist_directory` exists. If not, it raises a
        `FileNotFoundError`. Otherwise, it initializes and returns a `Chroma`
        vector store instance using the persisted data and the configured
        embedding function.

        Returns:
            A `langchain_chroma.Chroma` vector store instance.

        Raises:
            FileNotFoundError: If the `persist_directory` does not exist,
                               indicating that the vector store has not been
                               created yet.
        """
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                "Vector store not found. Please run indexer.py first."
            )

        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )
        return vector_store
