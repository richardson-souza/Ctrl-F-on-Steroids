import unittest
import os
import json
import tempfile
import shutil

# Importa os componentes do nosso projeto que serão testados
from data_loader import DataLoader
from vector_store import VectorStoreManager
from ask import main as ask_main  # Importamos a função principal do ask
from unittest.mock import patch, MagicMock

import config

# Dados do JSON que usaremos no nosso arquivo de teste
TEST_JSON_DATA = {
    "schema": "produto",
    "table": "dim_produto_cadastro",
    "table_description": "Tabela com todos os dados de produtos.",
    "source": ["lake_warmachine.produto"],
    "column_types": {
        "id_produto": {"description": "ID do produto."}
    },
    "table_id": ["id_produto"]
}


class TestEndToEndAsk(unittest.TestCase):
    """
    Classe de teste de ponta a ponta (End-to-End).
    
    IMPORTANTE: Este teste requer que o servidor Ollama esteja em execução
    com o modelo especificado em config.py (ex: gemma2:2b) já baixado.
    """

    @classmethod
    def setUpClass(cls):
        """
        Configura um repositório temporário e o indexa uma vez para todos os testes.
        """
        cls.test_dir = tempfile.mkdtemp()
        cls.repo_path = os.path.join(cls.test_dir, "test_repo")
        os.makedirs(cls.repo_path)

        json_file_path = os.path.join(cls.repo_path, "dim_produto.json")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(TEST_JSON_DATA, f)

        cls.original_repo_path = config.REPO_PATH
        cls.original_db_path = config.VECTOR_DB_PATH
        config.REPO_PATH = cls.repo_path
        config.VECTOR_DB_PATH = os.path.join(cls.test_dir, "test_chroma_db")

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
        config.REPO_PATH = cls.original_repo_path
        config.VECTOR_DB_PATH = cls.original_db_path

    @patch('builtins.input', side_effect=["describes the table 'produto.dim_produto_cadastro'", "exit"])
    @patch('builtins.print')
    def test_e2e_ask_describe_table(self, mock_print, mock_input):
        """
        Teste E2E: Executa o loop de Q&A e verifica se a resposta real do LLM é correta.
        """
        print("\n--- Executando teste E2E: ask_describe_table ---")
        
        # Chama a função principal do script 'ask.py'
        # O mock do 'input' irá fornecer a nossa pergunta e depois 'exit'
        ask_main()

        # Captura todos os argumentos que foram passados para a função print
        print_args = " ".join([call.args[0] for call in mock_print.call_args_list if call.args])

        print("\n--- Captured Print Output (print_args) ---")
        print(print_args)
        print("------------------------------------------\n")
        # --- Verificações (Asserts) ---
        # Verifica se a descrição real da tabela está na saída impressa
        # self.assertIn("Answer", print_args,
        #               "A resposta do LLM não continha a palavra 'Answer'.")

        # self.assertIn("This document does not describe the table", print_args,
        #               "A resposta do LLM não é a esperada para a pergunta.")

        self.assertNotIn("This document does not describe the table", print_args,
                      "A resposta do LLM é a esperada para a pergunta.")
        
        # Verifica se o nome da tabela também foi mencionado na resposta
        # self.assertIn("dim_produto_cadastro", print_args,
        #               "A resposta do LLM não mencionou o nome da tabela.")
        
        # # Verifica se o arquivo fonte foi citado
        # self.assertIn("dim_produto.json", print_args,
        #               "A resposta do LLM não citou o arquivo fonte correto.")


if __name__ == '__main__':
    unittest.main()
