import re
from typing import List
from langchain.docstore.document import Document
from sql_metadata.parser import Parser

from .base_processor import BaseProcessor


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
