from typing import List
from langchain.docstore.document import Document

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