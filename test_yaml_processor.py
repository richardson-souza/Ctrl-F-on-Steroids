import pytest
from data_loader import YamlProcessor


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


@pytest.fixture
def temp_yaml_file(tmpdir):
    """
    Cria um arquivo YAML temporário para ser usado nos testes.
    """
    file_path = tmpdir.join("test_dag_config.yaml")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(TEST_YAML_DATA)
    return str(file_path)


def test_yaml_processor_with_valid_dag(temp_yaml_file):
    processor = YamlProcessor()

    documents = processor.process(temp_yaml_file)

    expected_doc_count = 1 + 1 + 3 + 1
    assert len(documents) == expected_doc_count, (
        "O número de documentos gerados está incorreto."
    )

    page_contents = [doc.page_content for doc in documents]

    summary_doc_content = page_contents[1]
    assert (
        "describes the DAG configuration with ID 'fake_seller_dag'"
        in summary_doc_content
    )
    assert "owner is 'tester@example.com'" in summary_doc_content
    assert "runs on schedule: '0 5 * * 1'" in summary_doc_content

    process_data_task_doc = next(
        (p for p in page_contents if "task named 'process_data'" in p), None
    )
    assert process_data_task_doc is not None, (
        "Documento da tarefa 'process_data' não encontrado."
    )
    assert "uses the 'ecs' operator" in process_data_task_doc
    assert "runs the script 'src/fake_processor.py'" in process_data_task_doc

    execution_flow_doc = next(
        (p for p in page_contents if "execution flow for DAG" in p), None
    )
    assert execution_flow_doc is not None, (
        "Documento de fluxo de execução não encontrado."
    )
    assert "start_task >> process_data >> send_report" in execution_flow_doc
