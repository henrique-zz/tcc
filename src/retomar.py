from ultralytics import YOLO

def retomar_treinamento():
    print("🚑 Iniciando resgate do treinamento...")
    
    # Aponta para o arquivo last.pt da rodada que foi interrompida
    caminho_peso_salvo = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train\flamevision-yolov8n-run3\weights\last.pt"
    
    # Carrega o modelo com o estado exato de quando a luz caiu
    model = YOLO(caminho_peso_salvo)
    
    # O comando resume=True é mágico: ele já sabe o batch, as epochs e a pasta correta
    model.train(resume=True)
    
    print("✅ Resgate concluído!")

if __name__ == "__main__":
    retomar_treinamento()