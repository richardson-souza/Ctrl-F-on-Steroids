import pytest
import json
import chromadb


@pytest.fixture(scope="module")
def db_collection():
    """
    Initializes a fresh in-memory ChromaDB, processes and loads the data,
    and yields the collection object for tests to use.
    """
    client = chromadb.Client()  # In-memory
    collection_name = "test_collection"
    collection = client.get_or_create_collection(collection_name)

    # process_and_load_to_chroma(collection, "path/to/your/yaml/files") # PSEUDO-CODE
    # For this example, let's manually add some data
    # In a real scenario, you'd process your actual YAML files here.
    collection.add(
        ids=[
            "excelencia_seller_score:classificacao_seller",
            "excelencia_seller_score:__DAG_SUMMARY__",
        ],
        documents=[
            "The task classificacao_seller runs src/seller_classification.py",
            "The DAG excelencia_seller_score runs on schedule 0 3 * * *",
        ],
        metadatas=[
            {"doc_id": "excelencia_seller_score:classificacao_seller"},
            {"doc_id": "excelencia_seller_score:__DAG_SUMMARY__"},
        ],
    )

    print("\nChromaDB collection set up for testing.")
    yield collection

    # Teardown (optional, as it's in-memory)
    print("\nTearing down ChromaDB collection.")
    client.delete_collection(collection_name)


# -----------------
# 2. LOAD & PARAMETRIZE: Load the golden dataset and prepare for parametrization
# -----------------
def load_golden_dataset():
    with open("test/golden_dataset.json", "r") as f:
        return json.load(f)


# This decorator tells pytest to run the test function multiple times,
# once for each item in our golden dataset.
@pytest.mark.parametrize("test_case", load_golden_dataset())
def test_critical_document_retrieval(test_case, db_collection):
    """
    Tests that for a specific question, the EXACT expected document ID is retrieved.
    """
    question = test_case["question"]
    expected_ids = set(test_case["expected_ids"])
    K = 5  # Retrieve top 5

    results = db_collection.query(query_texts=[question], n_results=K)
    retrieved_ids = set(results["ids"][0])

    # This is a powerful assertion for critical knowledge
    assert expected_ids.issubset(retrieved_ids), (
        f"Failed to find all expected documents for question: '{question}'.\n"
        f"Expected: {expected_ids}\nGot: {retrieved_ids}"
    )


# -----------------
# 3. AGGREGATE METRICS TEST: Test the overall performance
# -----------------
def test_overall_retrieval_performance(db_collection):
    """
    Tests that the overall retrieval metrics for the entire golden dataset
    are above a defined threshold.
    """
    golden_dataset = load_golden_dataset()
    K = 5

    mrr_sum = 0
    hit_count = 0

    for item in golden_dataset:
        question = item["question"]
        ground_truth_ids = set(item["expected_ids"])

        results = db_collection.query(query_texts=[question], n_results=K)
        retrieved_ids = results["ids"][0]  # This is an ordered list

        # Calculate Hit
        if set(retrieved_ids).intersection(ground_truth_ids):
            hit_count += 1

        # Calculate Reciprocal Rank
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                mrr_sum += 1 / (i + 1)
                break

    num_queries = len(golden_dataset)
    hit_rate = hit_count / num_queries
    mrr = mrr_sum / num_queries

    print("\n--- Overall Metrics ---")
    print(f"Hit Rate: {hit_rate:.2f}")
    print(f"MRR: {mrr:.2f}")
    print("-----------------------")

    # Define and assert your quality bar
    MIN_HIT_RATE = 1.0
    MIN_MRR = 0.90

    assert hit_rate >= MIN_HIT_RATE, (
        f"Hit Rate {hit_rate:.2f} is below the threshold of {MIN_HIT_RATE}"
    )
    assert mrr >= MIN_MRR, f"MRR {mrr:.2f} is below the threshold of {MIN_MRR}"
