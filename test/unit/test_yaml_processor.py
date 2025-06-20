import pytest
from langchain.docstore.document import Document

from processor.yaml_processor import YamlProcessor

TEST_YAML_DATA = """
libraries: &commom_libraries
  maven:
  - "fake.library:spark-something_2.12:1.0.0"

dag:
  dag_id: "fake_seller_dag"
  schedule_interval: "0 5 * * 1"
  description: "A fake DAG for testing the YAML processor."
  catchup: False
  default_args:
    owner: "tester@example.com"
    start_date: "2024-01-01 00:00:00"

tasks:
  start_task:
    operator: "dummy"

  process_data:
    operator: "ecs"
    file: "src/fake_processor.py"
    task_definition: "fake-task-def"
  
  send_report:
    operator: "databricks"
    file: "src/fake_reporter.py"
    libraries: *commom_libraries

execution: 
  - "start_task >> process_data >> send_report"
"""

YAML_NO_DAG_CONTENT = """
config:
  setting1: value1
  setting2: value2
users:
  - name: Alice
  - name: Bob
"""

EMPTY_YAML_CONTENT = ""

INVALID_YAML_CONTENT = """
dag:
  dag_id: "broken_dag"
  tasks:
    task_1: operator: "dummy" # Invalid YAML: no space after colon
"""

MULTI_DAG_YAML_CONTENT = """
dag:
  dag_id: "dag_alpha"
  schedule_interval: "0 1 * * *"
  description: "First test DAG."
  default_args:
    owner: "owner_alpha@example.com"
tasks:
  alpha_task1:
    operator: "dummy"
  alpha_task2:
    operator: "python"
    file: "alpha.py"
execution:
  - "alpha_task1 >> alpha_task2"
---
dag:
  dag_id: "dag_beta"
  schedule_interval: "0 2 * * *"
  description: "Second test DAG."
  default_args:
    owner: "owner_beta@example.com"
tasks:
  beta_task1:
    operator: "bash"
  beta_task2:
    operator: "ecs"
    file: "beta_script.sh"
execution:
  - "beta_task1 >> beta_task2"
"""


@pytest.fixture
def yaml_processor_instance():
    return YamlProcessor()


@pytest.fixture
def create_temp_yaml_file(tmp_path):
    """Creates a temporary YAML file with given content for testing."""

    def _create_file(content, filename="test.yaml"):
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return str(file_path)

    return _create_file


def test_process_valid_dag_yaml(yaml_processor_instance, create_temp_yaml_file):
    """Tests processing a valid YAML file with a single Airflow DAG."""
    file_path = create_temp_yaml_file(TEST_YAML_DATA, "valid_dag.yaml")
    documents = yaml_processor_instance.process(file_path)

    assert len(documents) == 5, (
        "Incorrect number of documents generated for a valid DAG."
    )

    raw_doc = documents[0]
    assert raw_doc.page_content == TEST_YAML_DATA
    assert raw_doc.metadata["source"] == file_path
    assert raw_doc.metadata["doc_id"] == f"raw:{file_path}"

    summary_doc = documents[1]
    assert isinstance(summary_doc, Document)
    assert (
        "describes the DAG configuration with ID 'fake_seller_dag'"
        in summary_doc.page_content
    )
    assert "owner is 'tester@example.com'" in summary_doc.page_content
    assert "runs on schedule: '0 5 * * 1'" in summary_doc.page_content
    assert (
        "description is: 'A fake DAG for testing the YAML processor.'"
        in summary_doc.page_content
    )
    assert summary_doc.metadata["source"] == file_path
    assert summary_doc.metadata["dag_id"] == "fake_seller_dag"
    assert summary_doc.metadata["doc_id"] == "fake_seller_dag:__DAG_SUMMARY__"
    assert summary_doc.metadata["owner"] == "tester@example.com"

    page_contents = [doc.page_content for doc in documents]

    start_task_doc_content = next(
        (p for p in page_contents if "task named 'start_task'" in p), None
    )
    assert start_task_doc_content is not None, "Document for 'start_task' not found."
    assert "uses the 'dummy' operator" in start_task_doc_content
    assert (
        "runs the script 'not specified'" in start_task_doc_content
    )  # Default when no file
    assert (
        "It runs before the following task(s): process_data." in start_task_doc_content
    )
    assert (
        "It runs after the following task(s):" not in start_task_doc_content
    )  # No upstream

    process_data_task_doc = next(
        (p for p in page_contents if "task named 'process_data'" in p), None
    )
    assert process_data_task_doc is not None, "Document for 'process_data' not found."
    assert "uses the 'ecs' operator" in process_data_task_doc
    assert "runs the script 'src/fake_processor.py'" in process_data_task_doc
    assert "It runs after the following task(s): start_task." in process_data_task_doc
    assert "It runs before the following task(s): send_report." in process_data_task_doc

    send_report_task_doc = next(
        (p for p in page_contents if "task named 'send_report'" in p), None
    )
    assert send_report_task_doc is not None, "Document for 'send_report' not found."
    assert "uses the 'databricks' operator" in send_report_task_doc
    assert "runs the script 'src/fake_reporter.py'" in send_report_task_doc
    assert "It runs after the following task(s): process_data." in send_report_task_doc
    assert (
        "It runs before the following task(s):" not in send_report_task_doc
    )  # No downstream

    process_data_doc_obj = next(
        (d for d in documents if d.metadata.get("task_name") == "process_data"), None
    )
    assert process_data_doc_obj is not None
    assert process_data_doc_obj.metadata["source"] == file_path
    assert process_data_doc_obj.metadata["dag_id"] == "fake_seller_dag"
    assert process_data_doc_obj.metadata["doc_id"] == "fake_seller_dag:process_data"
    assert process_data_doc_obj.metadata["owner"] == "tester@example.com"
    assert process_data_doc_obj.metadata["operator"] == "ecs"


def test_process_yaml_with_no_dag(yaml_processor_instance, create_temp_yaml_file):
    """Tests processing a YAML file with no Airflow DAG definition."""
    file_path = create_temp_yaml_file(YAML_NO_DAG_CONTENT, "no_dag.yaml")
    documents = yaml_processor_instance.process(file_path)

    assert len(documents) == 1, "Should only generate raw content document."
    assert documents[0].page_content == YAML_NO_DAG_CONTENT
    assert documents[0].metadata["source"] == file_path


def test_process_empty_yaml_file(yaml_processor_instance, create_temp_yaml_file):
    """Tests processing an empty YAML file."""
    file_path = create_temp_yaml_file(EMPTY_YAML_CONTENT, "empty.yaml")
    documents = yaml_processor_instance.process(file_path)

    assert len(documents) == 1, (
        "Should generate one raw content document for an empty file."
    )
    assert documents[0].page_content == EMPTY_YAML_CONTENT
    assert documents[0].metadata["source"] == file_path


def test_process_invalid_yaml_file(
    yaml_processor_instance, create_temp_yaml_file, capsys
):
    """Tests processing an invalid YAML file and checks for a warning."""
    file_path = create_temp_yaml_file(INVALID_YAML_CONTENT, "invalid.yaml")
    documents = yaml_processor_instance.process(file_path)

    assert len(documents) >= 1
    assert documents[0].page_content == INVALID_YAML_CONTENT

    captured = capsys.readouterr()
    assert f"Warning: Invalid YAML format in {file_path}" in captured.out


def test_process_file_not_found(yaml_processor_instance, capsys):
    """Tests processing a non-existent file and checks for a warning."""
    non_existent_file = "non_existent_file.yaml"
    documents = yaml_processor_instance.process(non_existent_file)

    assert len(documents) == 0, "Should return an empty list for a non-existent file."
    captured = capsys.readouterr()
    assert f"Warning: Could not read file {non_existent_file}" in captured.out


def test_process_yaml_with_multiple_dags(
    yaml_processor_instance, create_temp_yaml_file
):
    """Tests processing a YAML file with multiple DAG definitions."""
    file_path = create_temp_yaml_file(MULTI_DAG_YAML_CONTENT, "multi_dag.yaml")
    documents = yaml_processor_instance.process(file_path)

    assert len(documents) == 7, "Incorrect number of documents for multi-DAG file."

    assert documents[0].page_content == MULTI_DAG_YAML_CONTENT

    dag_alpha_summary = next(
        (
            doc
            for doc in documents
            if doc.metadata.get("dag_id") == "dag_alpha"
            and "__DAG_SUMMARY__" in doc.metadata.get("doc_id", "")
        ),
        None,
    )
    assert dag_alpha_summary is not None
    assert "owner_alpha@example.com" in dag_alpha_summary.page_content

    alpha_task1_doc = next(
        (doc for doc in documents if doc.metadata.get("task_name") == "alpha_task1"),
        None,
    )
    assert alpha_task1_doc is not None
    assert "dag_id': 'dag_alpha'" in str(
        alpha_task1_doc.metadata
    )  # Check dag_id in metadata
    assert (
        "It runs before the following task(s): alpha_task2."
        in alpha_task1_doc.page_content
    )

    # Check for DAG Beta
    dag_beta_summary = next(
        (
            doc
            for doc in documents
            if doc.metadata.get("dag_id") == "dag_beta"
            and "__DAG_SUMMARY__" in doc.metadata.get("doc_id", "")
        ),
        None,
    )
    assert dag_beta_summary is not None
    assert "owner_beta@example.com" in dag_beta_summary.page_content

    beta_task2_doc = next(
        (doc for doc in documents if doc.metadata.get("task_name") == "beta_task2"),
        None,
    )
    assert beta_task2_doc is not None
    assert "dag_id': 'dag_beta'" in str(beta_task2_doc.metadata)
    assert "runs the script 'beta_script.sh'" in beta_task2_doc.page_content
    assert (
        "It runs after the following task(s): beta_task1."
        in beta_task2_doc.page_content
    )
