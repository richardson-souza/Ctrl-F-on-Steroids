import yaml
from typing import List
from langchain.docstore.document import Document

from .base_processor import BaseProcessor


class YamlProcessor(BaseProcessor):
    """
    Processes YAML files, with specialized handling for Airflow DAG definitions.

    This processor first loads the entire raw content of the YAML file as a
    `Document`. It then parses the YAML content, which may contain multiple
    YAML documents within a single file (separated by '---').

    If a parsed YAML document is identified as an Airflow DAG definition (typically
    a dictionary containing a top-level 'dag' key), this processor extracts
    detailed information and creates additional `Document` objects:

    1.  **DAG Summary Document**:
        A single `Document` summarizing the DAG's `dag_id`, `owner`,
        `schedule_interval`, and `description`.

    2.  **Individual Task Documents**:
        For each task defined within the DAG (e.g., under a 'tasks' key),
        a separate `Document` is created. This document details the task's
        name, the `operator` it uses, and the `file` or script it executes.
        If an 'execution' flow (e.g., "task_a >> task_b") is specified in
        the YAML, this task document will also include a textual description
        of its direct upstream and downstream task dependencies.

    The metadata for these specialized documents includes the source file path
    and relevant identifiers like 'dag_id', 'owner', and 'task_name' (for task
    documents), allowing for targeted retrieval.

    If a YAML document within the file does not conform to the expected Airflow
    DAG structure, or if the file is not an Airflow DAG definition at all,
    only the initial raw content `Document` (and any documents from other
    conforming DAGs within the same multi-document file) will be produced
    for that particular YAML document or file.

    Error Handling:
    If errors occur during file reading or YAML parsing (e.g., invalid YAML
    format), a warning is printed to the console, and the method returns
    any documents successfully processed up to that point. This could be an
    empty list if the error occurs very early, or just the raw content document
    if parsing fails later.
    """

    def _create_raw_document(self, file_path: str, raw_content: str) -> Document:
        """Creates a Document object for the raw file content."""
        raw_doc_id = f"raw:{file_path}"
        return Document(
            page_content=raw_content,
            metadata={"source": file_path, "doc_id": raw_doc_id},
        )

    def _parse_execution_flow(self, execution_flow_list: List[str]) -> dict:
        """Parses the execution flow strings to build task dependencies."""
        dependencies = {"upstream": {}, "downstream": {}}
        if isinstance(execution_flow_list, list):
            for flow in execution_flow_list:
                parts = [p.strip() for p in flow.split(">>")]
                for i in range(len(parts) - 1):
                    upstream_task, downstream_task = parts[i], parts[i + 1]
                    dependencies["downstream"].setdefault(upstream_task, []).append(
                        downstream_task
                    )
                    dependencies["upstream"].setdefault(downstream_task, []).append(
                        upstream_task
                    )
        return dependencies

    def _create_dag_summary_document(
        self,
        file_path: str,
        dag_id: str,
        owner: str,
        schedule: str,
        description: str,
    ) -> Document:
        """Creates a summary Document for a DAG."""
        summary_text = (
            f"This document describes the DAG configuration with ID '{dag_id}'. "
            f"The owner is '{owner}', it runs on schedule: '{schedule}', "
            f"and its description is: '{description}'."
        )
        summary_metadata = {
            "source": file_path,
            "dag_id": dag_id,
            "doc_id": f"{dag_id}:__DAG_SUMMARY__",
            "owner": owner,
        }
        return Document(page_content=summary_text, metadata=summary_metadata)

    def _create_task_document(
        self,
        file_path: str,
        dag_id: str,
        owner: str,
        task_name: str,
        task_details: dict,
        dependencies: dict,
    ) -> Document:
        """Creates a Document for a single DAG task."""
        operator = task_details.get("operator", "unknown")
        script = task_details.get("file", "not specified")
        task_text = (
            f"The DAG '{dag_id}' contains a task named '{task_name}'. "
            f"This task uses the '{operator}' operator and runs the script '{script}'."
        )

        if task_name in dependencies.get("upstream", {}):
            ups = ", ".join(dependencies["upstream"][task_name])
            task_text += f" It runs after the following task(s): {ups}."
        if task_name in dependencies.get("downstream", {}):
            downs = ", ".join(dependencies["downstream"][task_name])
            task_text += f" It runs before the following task(s): {downs}."

        task_metadata = {
            "source": file_path,
            "dag_id": dag_id,
            "task_name": task_name,
            "doc_id": f"{dag_id}:{task_name}",
            "owner": owner,
            "operator": operator,
        }
        return Document(page_content=task_text, metadata=task_metadata)

    def _process_dag_doc(self, doc_content: dict, file_path: str) -> List[Document]:
        """Processes a single YAML document identified as an Airflow DAG."""
        dag_documents = []
        dag_info = doc_content.get("dag", {})
        dag_id = dag_info.get("dag_id", "N/A")

        execution_flow_list = doc_content.get("execution", [])
        dependencies = self._parse_execution_flow(execution_flow_list)

        owner = dag_info.get("default_args", {}).get("owner", "N/A")
        schedule = dag_info.get("schedule_interval", "N/A")
        description = dag_info.get("description", "No description provided.")

        dag_documents.append(
            self._create_dag_summary_document(
                file_path, dag_id, owner, schedule, description
            )
        )

        tasks = doc_content.get("tasks", {})
        for task_name, task_details in tasks.items():
            dag_documents.append(
                self._create_task_document(
                    file_path,
                    dag_id,
                    owner,
                    task_name,
                    task_details,
                    dependencies,
                )
            )
        return dag_documents

    def process(self, file_path: str) -> List[Document]:
        """
        Loads and processes a YAML file, extracting raw content and detailed
        Airflow DAG information if present.

        Args:
            file_path: The absolute path to the YAML file.

        Returns:
            A list of `Document` objects. This list always includes a
            `Document` containing the raw content of the YAML file.
            If Airflow DAG definitions are found, it will also include:
            - A summary `Document` for each DAG.
            - A `Document` for each task within each DAG. Task documents
              will include descriptions of their upstream and downstream
              dependencies if an 'execution' flow is defined in the YAML.
            If an error occurs (e.g., invalid YAML), a warning is printed,
            and any documents successfully processed up to that point are returned.
            This could be an empty list if the error occurs before any processing.
        """
        documents = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
                documents.append(self._create_raw_document(file_path, raw_content))
                f.seek(0)
                yaml_data_list = list(yaml.safe_load_all(f))

            for yaml_doc_content in yaml_data_list:
                if isinstance(yaml_doc_content, dict) and "dag" in yaml_doc_content:
                    dag_specific_documents = self._process_dag_doc(
                        yaml_doc_content, file_path
                    )

                    # documents.append(dag_specific_documents)
                    documents.extend(dag_specific_documents)

        except yaml.YAMLError as e:
            print(f"Warning: Invalid YAML format in {file_path}. Error: {e}")
        except IOError as e:
            print(f"Warning: Could not read file {file_path}. Error: {e}")
        except Exception as e:
            print(
                f"Warning: Unexpected error processing YAML file {file_path}. Error: {e}"
            )
        return documents
