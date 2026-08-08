import yaml
from pathlib import Path

from src import treinamento

import src.treinamento as treinamento

def main():
    caminho_config = Path("configs/config.yaml")
    with open(caminho_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(".")
    pasta_destino = base_dir / config['dataset']['pasta_saida']
    pasta_runs = base_dir / "runs" / "train"

    print("\n============================================================")
    print("=== INICIANDO O TREINAMENTO DO MODELO YOLO =======")
    print("============================================================")
    treinamento.treinar_modelo(pasta_destino, pasta_runs, config['treinamento'])

if __name__ == "__main__":
    main()