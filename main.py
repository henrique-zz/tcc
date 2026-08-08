import yaml
from pathlib import Path

from src import processamento
from src import treinamento

import src.processamento as processamento
import src.treinamento as treinamento

def main():
    # 1. Carrega o arquivo de configuração de parâmetros
    caminho_config = Path("configs/config.yaml")
    if not caminho_config.exists():
        print("Erro: Arquivo config.yaml não foi encontrado na pasta raiz!")
        return

    with open(caminho_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Mapeia os diretórios locais com base no YAML
    base_dir = Path(".")
    pasta_origem = base_dir / config['dataset']['pasta_imagens']
    pasta_destino = base_dir / config['dataset']['pasta_saida']
    pasta_runs = base_dir / "runs" / "train"

    # Validação de segurança para garantir que as fotos originais estão no lugar certo
    if not pasta_origem.exists():
        print(f"Erro Crítico: A pasta '{pasta_origem.name}' não existe.")
        print(f"Por favor, crie a pasta '{pasta_origem.name}' e coloque as imagens originais nela.")
        return

    print("============================================================")
    print("=== PASSO 1: EXTRAÇÃO DE COMPONENTES E GERAÇÃO DE LABELS ===")
    print("============================================================")
    # Mescla os dicionários de parâmetros necessários para o processamento
    config_processamento = {**config['dataset'], **config['processamento']}
    processamento.preparar_dataset(pasta_origem, pasta_destino, config_processamento)

    print("\n============================================================")
    print("=== PASSO 2: INICIANDO O TREINAMENTO DO MODELO YOLO =======")
    print("============================================================")
    treinamento.treinar_modelo(pasta_destino, pasta_runs, config['treinamento'])

if __name__ == "__main__":
    main()