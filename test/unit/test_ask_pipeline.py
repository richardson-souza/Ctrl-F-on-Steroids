import unittest
import os
import json
import tempfile
import shutil

# Importa os componentes do nosso projeto que serão testados
from data_loader import DataLoader
from vector_store import VectorStoreManager
import config

# Dados do JSON que usaremos no nosso arquivo de teste
TEST_JSON_DATA = {
    "schema":"produto",
    "table":"dim_produto_cadastro",
    "table_description": "Tabela carregada com todos os atributos de produtos (id, nome, marca, etc.).",
    "source": ["lake_warmachine.produto"],
    "column_types":{
        "id_produto": { "description": "Identificação (ID) do produto." }
    },
    "table_id":["id_produto"]
}


class TestAskPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configura um repositório temporário e o indexa uma vez para todos os testes.
        """
        # Cria um diretório temporário para o nosso teste
        cls.test_dir = tempfile.mkdtemp()
        cls.repo_path = os.path.join(cls.test_dir, "test_repo")
        os.makedirs(cls.repo_path)

        # Cria o arquivo JSON de teste dentro do diretório temporário
        json_file_path = os.path.join(cls.repo_path, "dim_produto.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(TEST_JSON_DATA, f)

        # Sobrescreve temporariamente as configurações para apontar para o nosso ambiente de teste
        cls.original_repo_path = config.REPO_PATH
        cls.original_db_path = config.VECTOR_DB_PATH
        config.REPO_PATH = cls.repo_path
        config.VECTOR_DB_PATH = os.path.join(cls.test_dir, "test_chroma_db")

        # Executa o processo de indexação
        print("\n--- Configurando ambiente de teste e executando o indexador ---")
        loader = DataLoader(repo_path=config.REPO_PATH)
        documents = loader.load_repository_documents()
        vector_store_manager = VectorStoreManager()
        vector_store_manager.create_store_from_documents(documents)
        print("--- Índice de teste criado com sucesso ---")

    @classmethod
    def tearDownClass(cls):
        """
        Limpa o diretório temporário após a conclusão de todos os testes.
        """
        print("\n--- Desmontando ambiente de teste ---")
        shutil.rmtree(cls.test_dir)
        # Restaura as configurações originais
        config.REPO_PATH = cls.original_repo_path
        config.VECTOR_DB_PATH = cls.original_db_path

    def test_ask_describe_table_retrieval(self):
        """
        Teste de integração: Verifica se o retriever encontra o documento correto
        para uma pergunta sobre a descrição de uma tabela.
        """
        # --- Configura o retriever ---
        vector_store_manager = VectorStoreManager()
        retriever = vector_store_manager.get_retriever(
            search_type="mmr",
            search_kwargs={'k': 8}
        )

        # --- Faz a pergunta ---
        query = "describes the table 'produto.dim_produto_cadastro'"
        
        # Chama o retriever diretamente para obter os documentos relevantes
        retrieved_docs = retriever.invoke(query)
        
        # --- Verificações (Asserts) ---
        
        # Verifica se o retriever retornou algum documento
        self.assertGreater(len(retrieved_docs), 0, "O retriever não retornou nenhum documento.")
        
        # Verifica se um dos documentos retornados é o nosso "documento de resumo" inteligente
        found_summary = False
        for doc in retrieved_docs:
            # Verifica se o conteúdo do documento é o resumo que esperamos
            if "describes the table 'produto.dim_produto_cadastro'" in doc.page_content and \
               "Description: Tabela carregada com todos os atributos" in doc.page_content:
                found_summary = True
                break
            
        self.assertTrue(found_summary, "O documento de resumo correto não foi encontrado pelo retriever.")


if __name__ == '__main__':
    unittest.main()
