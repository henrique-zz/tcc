import yaml
from ultralytics import YOLO
from pathlib import Path

def treinar_modelo(pasta_dataset: Path, pasta_runs: Path, config: dict):
    """
    Função genérica para carregar o modelo YOLO e iniciar o treinamento
    com os parâmetros definidos no config.yaml.
    """
    print(f"Carregando o modelo base: {config['modelo_base']}...")
    model = YOLO(config['modelo_base'])
    
    # Caminho do arquivo de mapeamento do dataset
    arquivo_yaml = pasta_dataset / "dataset.yaml"
    
    print("Disparando o treinamento na Placa de Vídeo...")
    model.train(
        data=arquivo_yaml.resolve().as_posix(),
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        name=config['nome_experimento'],
        project=pasta_runs.resolve().as_posix(),
        device=config['device'],
        workers=config.get('workers', 0)
    )
    print("Processo de treinamento concluído!")

if __name__ == "__main__":
    caminho_config = Path("configs/config.yaml")
    
    with open(caminho_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Ele vai pegar o caminho do FlameVision ou FLAME que você colocou no config.yaml
    pasta_dataset = Path(config['dataset']['pasta_saida'])
    pasta_runs = Path("runs/train")
    
    print("\nIniciando treinamento direto...")
    treinar_modelo(pasta_dataset, pasta_runs, config['treinamento'])