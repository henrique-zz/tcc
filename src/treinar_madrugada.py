from ultralytics import YOLO

def treinar_madrugada():
    dataset_path = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\datasets\flamevision-dataset\Detection\dataset.yaml"
    
    # Colocando o caminho ABSOLUTO aqui, o YOLO é obrigado a respeitar e não cria a pasta "detect"
    pasta_destino = r"C:\Users\Henrique S\Downloads\tcc\tcc-refatorado\runs\train"
    
    for i in range(4, 11):
        print(f"\n{'='*50}")
        print(f"🔥 INICIANDO RODADA {i} DE 10 🔥")
        print(f"{'='*50}\n")
        
        model = YOLO("pesos/yolov8n.pt")
        nome_pasta = f"flamevision-yolov8n-run{i}"
        
        model.train(
            data=dataset_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,
            workers=0,
            project=pasta_destino, # Passando o caminho absoluto
            name=nome_pasta
        )
        
        print(f"\n✅ Rodada {i} salva com sucesso em {pasta_destino}\\{nome_pasta}\n")

if __name__ == "__main__":
    treinar_madrugada()