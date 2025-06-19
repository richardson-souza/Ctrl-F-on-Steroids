import json
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain.docstore.document import Document

from base_processor import BaseProcessor


class JsonProcessor(BaseProcessor):
    """
    Processes JSON files, extracting raw content and structured data.

    This processor is specifically designed to handle JSON files that may
    represent data dictionaries. If a JSON file contains 'schema' and 'table'
    keys, it will generate additional `Document` objects:
    - A summary document for the table, including its description.
    - Individual documents for each column, detailing their descriptions.

    If the JSON file does not conform to this expected data dictionary
    structure, it will still load the raw content of the JSON file as a
    single `Document`.

    Attributes:
        None
    """

    def process(self, file_path: str) -> List[Document]:
        """
        Loads and processes a JSON file.

        Args:
            file_path: The absolute path to the JSON file.

        Returns:
            A list of `Document` objects. This list includes:
            - A `Document` containing the raw JSON content.
            - If the JSON is a data dictionary:
                - A `Document` summarizing the table.
                - A `Document` for each column in the table.
            Returns an empty list if an error occurs during processing or if
            the JSON is a data dictionary but lacks 'schema' or 'table' keys
            after loading the raw content.
        """
        documents = []
        try:
            raw_documents = TextLoader(file_path, encoding="utf-8").load()
            documents.extend(raw_documents)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "schema" not in data or "table" not in data:
                return documents

            table_name = f"{data.get('table', '')}"
            schema = f"{data.get('schema', '')}"
            table_metadata = {
                "source": file_path,
                "table": table_name,
                "schema": schema,
            }

            summary_text = f"This document describes the table '{table_name}' in the schema '{schema}'. Description: {data.get('table_description', 'N/A')}"
            documents.append(
                Document(page_content=summary_text, metadata=table_metadata)
            )

            for col_name, col_data in data.get("column_types", {}).items():
                col_metadata = table_metadata.copy()
                col_metadata["column"] = col_name
                documents.append(
                    Document(
                        page_content=f"In table '{table_name}', column '{col_name}' has the description: {col_data.get('description', 'N/A')}",
                        metadata=col_metadata,
                    )
                )
        except json.JSONDecodeError as e:
            print(
                f"Warning: Could not parse JSON file {file_path}. Invalid JSON. Error: {e}"
            )
        except Exception as e:
            print(f"Warning: Could not process JSON file {file_path}. Error: {e}")
        return documents
