import ast
from typing import List
from langchain.docstore.document import Document

from base_processor import BaseProcessor


class PythonProcessor(BaseProcessor):
    """
    Processes .py files using Abstract Syntax Tree (AST) parsing.

    This processor loads the raw content of a Python file and then uses
    the `ast` module to parse its structure. It generates several types
    of `Document` objects:
    1.  **Raw Content**: A document containing the entire source code of
        the Python file.
    2.  **Import Summary**: A document summarizing all libraries and modules
        imported by the script.
    3.  **Function Definitions**: A document for each function defined in
        the script, including the function's name and its docstring (or
        "No docstring provided." if none exists).

    The metadata for these specialized documents includes the source file path,
    a 'content_type' for summaries, and 'function_name' for function documents.

    Inherits from `BaseProcessor`.
    """

    def process(self, file_path: str) -> List[Document]:
        """
        Loads and processes a Python file using AST parsing.

        Args:
            file_path: The absolute path to the Python file.

        Returns:
            A list of `Document` objects. This list includes:
            - A `Document` with the raw Python source code.
            - A `Document` summarizing imported libraries.
            - A `Document` for each function definition, containing its
              name and docstring.
            Returns an empty list if an error occurs during file reading
            or AST parsing.
        """
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                python_content = f.read()
            documents.append(
                Document(page_content=python_content, metadata={"source": file_path})
            )

            tree = ast.parse(python_content)
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imports.update(
                {
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                }
            )

            summary_text = f"This Python script imports the following libraries: {', '.join(sorted(imports))}."
            documents.append(
                Document(
                    page_content=summary_text,
                    metadata={"source": file_path, "content_type": "python_summary"},
                )
            )

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node) or "No docstring provided."
                    func_text = f"Function '{node.name}': {docstring}"
                    documents.append(
                        Document(
                            page_content=func_text,
                            metadata={"source": file_path, "function_name": node.name},
                        )
                    )
        except Exception as e:
            print(f"Warning: Could not process Python file {file_path}. Error: {e}")
        return documents
