import pytest
import json
from data_loader import JsonProcessor

TEST_JSON_DATA = {
    "schema": "fakeschema",
    "table": "mock_table",
    "operator": "test_operator",
    "api": None,
    "bu": "TestBU",
    "table_description": "This is a fake table description for testing purposes.",
    "source": ["fake_source.table1", "fake_source.table2"],
    "column_types": {
        "pk_id_seller": {
            "column_type": "int",
            "max_length": "None",
            "description": "A fake ID for a fake seller.",
            "is_metric": False,
            "is_nullable": False,
            "pii_confidentiality_impact_level": "low_sensibility",
            "fk_source": None,
        },
        "regra_score": {
            "column_type": "string",
            "max_length": "25",
            "description": "A fake rule for a fake score, such as 'Test-30' or 'Test-180'.",
            "is_metric": False,
            "is_nullable": False,
            "pii_confidentiality_impact_level": "low_sensibility",
            "fk_source": None,
        },
        "regra_score_num": {
            "column_type": "int",
            "max_length": "None",
            "description": "A numeric fake rule for a fake score, like 30, 180, or 0.",
            "is_metric": False,
            "is_nullable": False,
            "pii_confidentiality_impact_level": "low_sensibility",
            "fk_source": None,
        },
        "data_processamento": {
            "column_type": "string",
            "max_length": "19",
            "description": "A fake processing date in YYYY-MM-DD HH:MI:SS format.",
            "is_metric": False,
            "is_nullable": False,
            "pii_confidentiality_impact_level": "low_sensibility",
            "fk_source": None,
        },
    },
    "table_id": ["pk_id_seller"],
    "sgbd": "custom",
}


@pytest.fixture
def temp_json_file(tmpdir):
    file_path = tmpdir.join("test_data_dictionary.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(TEST_JSON_DATA, f, ensure_ascii=False, indent=4)

    return str(file_path)


def test_json_processor_with_valid_data(temp_json_file):
    processor = JsonProcessor()
    documents = processor.process(temp_json_file)
    num_columns = len(TEST_JSON_DATA["column_types"])
    expected_doc_count = 1 + 1 + num_columns
    assert len(documents) == expected_doc_count, (
        "The number of generated documents is incorrect."
    )

    page_contents = [doc.page_content for doc in documents]

    summary_doc_content = page_contents[1]
    assert "describes the table 'mock_table'" in summary_doc_content
    assert (
        "This is a fake table description for testing purposes." in summary_doc_content
    )

    pk_id_seller_doc_content = page_contents[2]
    assert "In table 'mock_table', column 'pk_id_seller'" in pk_id_seller_doc_content
    assert "A fake ID for a fake seller." in pk_id_seller_doc_content

    data_proc_doc_content = page_contents[5]
    assert "In table 'mock_table', column 'data_processamento'" in data_proc_doc_content
    assert "A fake processing date" in data_proc_doc_content
