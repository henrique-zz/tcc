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
        device=config['device']
    )
    print("Processo de treinamento concluído!")