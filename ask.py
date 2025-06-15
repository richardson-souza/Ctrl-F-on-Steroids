import os
from langchain_ollama import OllamaLLM
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from vector_store import VectorStoreManager
import config


def main():
    vector_store_manager = VectorStoreManager()

    if not os.path.exists(config.VECTOR_DB_PATH):
        print(f"Vector database not found at '{config.VECTOR_DB_PATH}'.")
        print("Please run `python indexer.py` first to create the database.")
        return

    try:
        vector_store = vector_store_manager.get_vector_store()
    except Exception as e:
        print(f"An error occurred while loading the vector store: {e}")
        return

    llm = OllamaLLM(model=config.LLM_MODEL, temperature=0)

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The absolute file path of the source document.",
            type="string",
        ),
        AttributeInfo(
            name="dag_id",
            description="The unique identifier for an Airflow DAG.",
            type="string",
        ),
        AttributeInfo(
            name="owner",
            description="The owner or creator of a DAG, usually an email address.",
            type="string",
        ),
        AttributeInfo(
            name="schedule",
            description="The cron schedule for a DAG, e.g., '0 5 * * 1'.",
            type="string",
        ),
        AttributeInfo(
            name="task_name",
            description="The name of a specific task within a DAG.",
            type="string",
        ),
        AttributeInfo(
            name="operator",
            description="The type of operator used by a DAG task, e.g., 'ecs' or 'databricks'.",
            type="string",
        ),
        AttributeInfo(
            name="dag_description",
            description="The description of an Airflow DAG.",
            type="string",
        ),
        AttributeInfo(
            name="script_file",
            description="The name of the script file executed by a task, e.g., 'main.py'.",
            type="string",
        ),
        AttributeInfo(
            name="schema",
            description="The name of a database schema.",
            type="string",
        ),
        AttributeInfo(
            name="table",
            description="The name of a database table.",
            type="string",
        ),
        AttributeInfo(
            name="column",
            description="The name of a column in a database table.",
            type="string",
        ),
        AttributeInfo(
            name="table_description",
            description="The description of a database table.",
            type="string",
        ),
        AttributeInfo(
            name="content_type",
            description="The type of a column in a database table.",
            type="string",
        ),
        AttributeInfo(
            name="content_type",
            description="The of a column in a database table.",
            type="string",
        ),
        AttributeInfo(
            name="function_name",
            description="The name of a Python function.",
            type="string",
        ),
    ]

    document_content_description = "The documents contain summaries and raw content from a data engineering codebase. The content includes details about Airflow DAGs (like owner, schedule, and description from dag.yaml files), database tables (like columns and sources from JSON data dictionaries), and the full text of SQL and Python scripts."

    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True,
        enable_limit=True,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    print("✅ Q&A Tool is ready. Ask questions about your codebase.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        if query:
            try:
                response = qa_chain.invoke(query)
                print("\n--- Answer ---")
                print(response["result"].strip())
                print("\n--- Sources ---")
                source_files = sorted(
                    list(
                        {doc.metadata["source"] for doc in response["source_documents"]}
                    )
                )
                for source in source_files:
                    print(f"- {source}")
            except Exception as e:
                print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
