import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document

from processor.yaml_processor import YamlProcessor
from processor.json_processor import JsonProcessor
from processor.sql_processor import SqlProcessor
from processor.python_processor import PythonProcessor

import config


class BaseProcessor:
    """
    Abstract base class for file processors.

    This class defines the interface for processing different file types.
    Subclasses are expected to implement the `process` method to handle
    the specifics of reading a file and converting its content into a list
    of Langchain `Document` objects.

    The `process` method in this base class raises a `NotImplementedError`
    to ensure that any concrete subclass provides its own implementation.
    """

    def process(self, file_path: str) -> List[Document]:
        raise NotImplementedError(
            "The process() method must be implemented by the subclass."
        )


class DataLoader:
    """
    Handles loading and processing of different file types from a repository,
    delegating to specialized processor classes.

    This class walks through a specified repository directory, identifies files
    with allowed extensions, and uses a dispatch mechanism to route each file
    to an appropriate `BaseProcessor` subclass (e.g., `JsonProcessor`,
    `SqlProcessor`).

    If a specialized processor is not available for a given file extension,
    it falls back to using `TextLoader` and `RecursiveCharacterTextSplitter`
    to load and chunk the raw text content of the file.

    Attributes:
        repo_path (str): The absolute path to the root of the code repository
                         to be processed.
        dispatch (dict): A dictionary mapping file extensions (e.g., '.json')
                         to instances of their corresponding processor classes.
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.dispatch = {
            # ".json": JsonProcessor(),
            ".yaml": YamlProcessor(),
            # ".sql": SqlProcessor(),
            # ".py": PythonProcessor(),
        }

    def _load_and_split_documents(self, file_path: str) -> List[Document]:
        """
        Delegates file processing to an appropriate specialized processor or
        uses a generic text loader if no specialized processor is found.

        For recognized file extensions ('.json', '.yaml', '.sql', '.py'), this
        method calls the `process` method of the corresponding specialized
        processor.

        For other file extensions listed in `config.ALLOWED_EXTENSIONS` but
        not having a dedicated processor, it loads the file as plain text
        using `TextLoader` and then splits it into chunks using
        `RecursiveCharacterTextSplitter`.

        Args:
            file_path: The absolute path to the file to be processed.

        Returns:
            A list of `Document` objects generated from the file. Returns an
            empty list if processing fails or if the file type is not supported
            and cannot be loaded as text.
        """
        file_extension = os.path.splitext(file_path)[1]

        processor = self.dispatch.get(file_extension)
        if processor:
            return processor.process(file_path)
        else:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                return text_splitter.split_documents(documents)
            except Exception:
                return []

    def load_repository_documents(self) -> List[Document]:
        """
        Walks the repository path, processes all allowed file types, and
        aggregates the resulting `Document` objects.

        It iterates through all files in the `self.repo_path` directory (and
        its subdirectories). For each file, it checks if its extension is
        present in `config.ALLOWED_EXTENSIONS`. If so, it calls
        `_load_and_split_documents` to process the file.

        Returns:
            A list containing all `Document` objects generated from all
            processed files in the repository.
        """
        all_documents = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if any(file.endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    docs = self._load_and_split_documents(file_path)
                    if docs:
                        all_documents.extend(docs)
        print(f"Found and processed {len(all_documents)} document chunks.")
        return all_documents
