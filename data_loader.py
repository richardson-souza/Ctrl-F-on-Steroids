import os
import json
import yaml
import re
import ast
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.docstore.document import Document
from sql_metadata.parser import Parser

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


class YamlProcessor(BaseProcessor):
    """
    Processes YAML files, with specialized handling for Airflow DAG definitions.

    This processor first loads the entire raw content of the YAML file as a
    `Document`. It then parses the YAML content, which may contain multiple
    YAML documents within a single file.

    If a parsed YAML document appears to be an Airflow DAG definition (i.e.,
    it's a dictionary containing a 'dag' key), this processor will extract
    detailed information and create additional `Document` objects for:
    1.  **DAG Summary**: A document summarizing the DAG's ID, owner, schedule,
        and description.
    2.  **Individual Tasks**: A document for each task defined within the DAG,
        detailing the task's name, operator, and associated script/file.
    3.  **Execution Flow**: A document outlining the defined execution order or
        dependencies of tasks, if specified.

    The metadata for these specialized documents includes the source file path
    and relevant identifiers like 'dag_id' and 'task_name'.

    If the YAML file or a document within it does not conform to the expected
    Airflow DAG structure, only the raw content document (and summaries for
    any conforming DAGs in the same file) will be produced for that content.

    Inherits from `BaseProcessor`.
    """

    def process(self, file_path: str) -> List[Document]:
        """
        Loads and processes a YAML file, extracting raw content and detailed
        Airflow DAG information if present.

        Args:
            file_path: The absolute path to the YAML file.

        Returns:
            A list of `Document` objects. The list will always include a
            document containing the raw content of the YAML file.
            If Airflow DAG definitions are found, it will also include:
            - A summary `Document` for each DAG.
            - A `Document` for each task within each DAG.
            - A `Document` for the execution flow of each DAG, if specified.
            If an error occurs (e.g., invalid YAML), a warning is printed,
            and any documents successfully processed up to that point are returned.
            This could be an empty list if the error occurs very early.
        """
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
                documents.append(
                    Document(page_content=raw_content, metadata={"source": file_path})
                )
                f.seek(0)
                yaml_docs = list(yaml.safe_load_all(f))

            for doc in yaml_docs:
                if isinstance(doc, dict) and "dag" in doc:
                    dag_info = doc.get("dag", {})
                    dag_id = dag_info.get("dag_id", "N/A")
                    source_metadata = {"source": file_path, "dag_id": dag_id}

                    owner = dag_info.get("default_args", {}).get("owner", "N/A")
                    schedule = dag_info.get("schedule_interval", "N/A")
                    dag_description = dag_info.get(
                        "description", "No description provided."
                    )
                    summary_text = f"This document describes the DAG configuration with ID '{dag_id}'. The owner is '{owner}', it runs on schedule: '{schedule}', and its description is: '{dag_description}'."
                    documents.append(
                        Document(page_content=summary_text, metadata=source_metadata)
                    )

                    tasks = doc.get("tasks", {})
                    for task_name, task_details in tasks.items():
                        operator = task_details.get("operator", "unknown")
                        script = task_details.get("file", "not specified")
                        task_text = f"The DAG '{dag_id}' contains a task named '{task_name}'. This task uses the '{operator}' operator and runs the script '{script}'."
                        task_metadata = source_metadata.copy()
                        task_metadata["task_name"] = task_name
                        documents.append(
                            Document(page_content=task_text, metadata=task_metadata)
                        )

                    execution_flow = doc.get("execution", [])
                    if execution_flow:
                        flow_str = (
                            ", ".join(execution_flow)
                            if isinstance(execution_flow, list)
                            else str(execution_flow)
                        )
                        execution_text = f"The execution flow for DAG '{dag_id}' is defined as follows: {flow_str}."
                        documents.append(
                            Document(
                                page_content=execution_text, metadata=source_metadata
                            )
                        )

        except Exception as e:
            print(f"Warning: Could not process YAML file {file_path}. Error: {e}")
        return documents


class SqlProcessor(BaseProcessor):
    """
    Processes .sql files, extracting raw content and a summary of tables used.

    This processor first loads the entire raw content of the SQL file as a
    `Document`. It then attempts to identify the main query within the SQL
    script, particularly looking for common patterns like `CREATE TABLE AS`,
    `CREATE VIEW AS`, `CREATE MATERIALIZED VIEW AS`, or Redshift `UNLOAD`
    statements.

    If a main query is successfully extracted, it uses the `sql-metadata`
    library to parse this query and identify the source tables. A summary
    `Document` is then created, listing these tables.

    The special comment "-- Databricks notebook source" is removed from the
    beginning of the SQL content if present.

    Inherits from `BaseProcessor`.
    """

    def process(self, file_path: str) -> List[Document]:
        """
        Loads and processes a SQL file.

        Args:
            file_path: The absolute path to the SQL file.

        Returns:
            A list of `Document` objects. This list includes:
            - A `Document` containing the raw SQL content (with Databricks
              comments removed).
            - If a main query can be parsed, a `Document` summarizing the
              tables read by that query.
            Returns a list containing only the raw content document if parsing
            fails or if no specific query pattern is matched for summarization.
            Returns an empty list if an error occurs during initial file reading.
        """
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = (
                    f.read().replace("-- Databricks notebook source", "").strip()
                )
            documents.append(
                Document(page_content=sql_content, metadata={"source": file_path})
            )

            sql_for_parsing = sql_content
            unload_match = re.search(
                r"unload\s*\(\s*\$\$(.*?)\$\$\s*\)",
                sql_content,
                re.DOTALL | re.IGNORECASE,
            )
            create_mview_match = re.search(
                r"create\s+materialized\s+view\s+.*?\s+as\s+(.*)",
                sql_content,
                re.DOTALL | re.IGNORECASE,
            )
            create_view_match = re.search(
                r"create\s+(or\s+replace\s+)?view\s+.*?\s+as\s+(.*)",
                sql_content,
                re.DOTALL | re.IGNORECASE,
            )
            create_table_match = re.search(
                r"create\s+(temp\s+)?table\s+.*?as\s*(\((.*?)\)|(.*))",
                sql_content,
                re.DOTALL | re.IGNORECASE,
            )

            if unload_match:
                sql_for_parsing = unload_match.group(1).strip()
            elif create_mview_match:
                sql_for_parsing = create_mview_match.group(1).strip()
            elif create_view_match:
                sql_for_parsing = create_view_match.group(2).strip()
            elif create_table_match:
                sql_for_parsing = (
                    create_table_match.group(3) or create_table_match.group(4) or ""
                ).strip()

            try:
                parser = Parser(sql_for_parsing)
                tables_str = ", ".join(parser.tables) if parser.tables else "none"
                summary_text = (
                    f"This SQL script reads from the following tables: {tables_str}."
                )
                documents.append(
                    Document(
                        page_content=summary_text,
                        metadata={"source": file_path, "content_type": "sql_summary"},
                    )
                )
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Could not process SQL file {file_path}. Error: {e}")
        return documents


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
            ".json": JsonProcessor(),
            ".yaml": YamlProcessor(),
            ".sql": SqlProcessor(),
            ".py": PythonProcessor(),
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
