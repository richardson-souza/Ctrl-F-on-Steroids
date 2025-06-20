import unittest
import json
import os
import tempfile
from data_loader import DataLoader
from langchain.docstore.document import Document

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        """
        Configura o ambiente de teste antes de cada teste.
        Cria um arquivo JSON temporário.
        """
        # Dados do JSON fornecidos pelo usuário
        self.json_data = {
            "schema":"produto",
            "table":"dim_produto_cadastro",
            "operator": "operator_ecs",
            "api": None,
            "bu": "produto",
            "table_description": "Tabela carregada com todos os atributos de produtos.",
            "source": [
                "lake_warmachine.produto",
                "lake_warmachine.seller"
            ],
            "column_types":{
                "id_produto":{
                    "column_type":"int",
                    "max_length":"None",
                    "description":"Identificação (ID) do produto.",
                    "is_metric":False,
                    "is_nullable":True,
                    "pii_confidentiality_impact_level":None
                },               
                "nome_produto":{
                    "column_type":"string",
                    "max_length":"200",
                    "description":"Nome do produto.",
                    "is_metric":False,
                    "is_nullable":True,
                    "pii_confidentiality_impact_level":"high_sensibility"
                }
            },
            "table_id":["id_produto","id_seller"]
        }
        
        # Cria um arquivo temporário para o teste
        # O NamedTemporaryFile nos dá um nome de arquivo que podemos usar
        # O 'delete=False' garante que o arquivo não seja excluído ao fechar,
        # para que nosso método possa abri-lo.
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding='utf-8')
        json.dump(self.json_data, self.temp_file)
        self.temp_file_path = self.temp_file.name
        self.temp_file.close()

        # Instancia o DataLoader (o repo_path não é usado por este método específico)
        self.loader = DataLoader(repo_path=".")

    def tearDown(self):
        """
        Limpa o ambiente após cada teste.
        Remove o arquivo JSON temporário.
        """
        os.remove(self.temp_file_path)

    def test_process_custom_json(self):
        """
        Testa o pré-processamento de um arquivo JSON de dicionário de dados.
        """
        # Chama o método que queremos testar
        documents = self.loader._process_custom_json(self.temp_file_path)

        # --- Asserções (Verificações) ---

        # 1. Verifica a quantidade total de documentos gerados
        #    Deveria ser 1 (conteúdo bruto) + 1 (resumo da tabela) + 2 (colunas) = 4
        num_columns = len(self.json_data["column_types"])
        expected_doc_count = 1 + 1 + num_columns
        self.assertEqual(len(documents), expected_doc_count)

        # 2. Verifica o conteúdo do primeiro documento (conteúdo bruto)
        with open(self.temp_file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        self.assertEqual(documents[0].page_content, raw_content)
        self.assertEqual(documents[0].metadata["source"], self.temp_file_path)

        # 3. Verifica o conteúdo do segundo documento (resumo da tabela)
        summary_doc = documents[1]
        self.assertIn("describes the table 'produto.dim_produto_cadastro'", summary_doc.page_content)
        self.assertIn("Description: Tabela carregada com todos os atributos de produtos.", summary_doc.page_content)
        self.assertIn("primary key(s) are: id_produto, id_seller", summary_doc.page_content)
        self.assertIn("sourced from the following locations: lake_warmachine.produto, lake_warmachine.seller", summary_doc.page_content)
        self.assertEqual(summary_doc.metadata["table"], "produto.dim_produto_cadastro")

        # 4. Verifica o conteúdo do documento da primeira coluna (id_produto)
        id_produto_doc = documents[2]
        self.assertIn("In table 'produto.dim_produto_cadastro', the column 'id_produto'", id_produto_doc.page_content)
        self.assertIn("type 'int'", id_produto_doc.page_content)
        self.assertIn("Description: Identificação (ID) do produto.", id_produto_doc.page_content)
        self.assertNotIn("PII", id_produto_doc.page_content) # PII é nulo, não deve aparecer
        self.assertEqual(id_produto_doc.metadata["column"], "id_produto")
        
        # 5. Verifica o conteúdo do documento da segunda coluna (nome_produto)
        nome_produto_doc = documents[3]
        self.assertIn("In table 'produto.dim_produto_cadastro', the column 'nome_produto'", nome_produto_doc.page_content)
        self.assertIn("type 'string'", nome_produto_doc.page_content)
        self.assertIn("max length of '200'", nome_produto_doc.page_content)
        self.assertIn("impact level of 'high_sensibility'", nome_produto_doc.page_content) # Verifica a menção de PII
        self.assertEqual(nome_produto_doc.metadata["column"], "nome_produto")


if __name__ == '__main__':
    unittest.main()
