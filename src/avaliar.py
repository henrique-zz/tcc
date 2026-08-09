import yaml
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
from ultralytics import YOLO

def avaliar_modelo():
    # 1. Lê o arquivo de configurações para saber onde estão as pastas
    caminho_config = Path("configs/config.yaml")
    if not caminho_config.exists():
        print("Erro: Arquivo config.yaml não encontrado.")
        return

    with open(caminho_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(".")
    pasta_dataset = base_dir / config['dataset']['pasta_saida']
    nome_experimento = config['treinamento']['nome_experimento']
    
    # Define os caminhos baseados nas configurações
    pasta_treino = base_dir / "runs" / "train" / nome_experimento
    pasta_predict = base_dir / "runs" / "predict"
    caminho_pesos = pasta_treino / "weights" / "best.pt"
    
    if not caminho_pesos.exists():
        print(f"Erro: Arquivo 'best.pt' não encontrado em {caminho_pesos}")
        print("Aguarde o término do treinamento antes de avaliar!")
        return

    # 2. Carrega a IA treinada
    print(f"Carregando o modelo treinado: {caminho_pesos}...")
    model = YOLO(caminho_pesos.resolve().as_posix())

    # 3. Predição nas imagens de teste (A prova final)
    print("\nGerando predições visuais...")
    pasta_teste = pasta_dataset / "images" / "test"
    model.predict(
        source=pasta_teste.resolve().as_posix(),
        save=True,
        save_txt=True,
        save_conf=True,
        name=nome_experimento,
        project=pasta_predict.resolve().as_posix(),
    )

    print("Arrumando a bagunça do YOLO (movendo imagens para a pasta 'images')...")
    pasta_resultados = pasta_predict / nome_experimento
    pasta_imagens_preditas = pasta_resultados / "images"
    pasta_imagens_preditas.mkdir(parents=True, exist_ok=True)

    # Procura todas as imagens que o YOLO jogou soltas e move para a pasta images
    for arquivo in pasta_resultados.glob("*.*"):
        if arquivo.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            shutil.move(str(arquivo), str(pasta_imagens_preditas / arquivo.name))

    # 4. Cálculo das Métricas
    print("\nCalculando as métricas da IA...")
    arquivo_yaml = pasta_dataset / "dataset.yaml"
    metrics = model.val(
        data=arquivo_yaml.resolve().as_posix(),
        split="test",
    )

    # Monta o texto bonitinho das métricas
    texto_metricas = (
        f"{'=' * 40}\n"
        f"MÉTRICAS NO CONJUNTO DE TESTE\n"
        f"{'=' * 40}\n"
        f"  mAP50     : {metrics.box.map50:.4f}\n"
        f"  mAP50-95  : {metrics.box.map:.4f}\n"
        f"  Precision : {metrics.box.mp:.4f}\n"
        f"  Recall    : {metrics.box.mr:.4f}\n"
        f"{'=' * 40}\n"
    )

    # Imprime no terminal para você ver na hora
    print("\n" + texto_metricas)

    # Cria e salva o arquivo .txt na mesma pasta das imagens de teste
    caminho_salvar = pasta_predict / nome_experimento / "metricas" / "metricas_finais.txt"
    
    # Garante que a pasta existe antes de salvar
    caminho_salvar.parent.mkdir(parents=True, exist_ok=True)
    
    with open(caminho_salvar, "w", encoding="utf-8") as f:
        f.write(texto_metricas)
        
    print(f"✅ Arquivo salvo com sucesso em: {caminho_salvar}")

    # 5. Visual Check (Lado a Lado)
    print("\nAbrindo janela de comparação visual (Real vs Predito)...")
    test_label_dir = pasta_dataset / "labels" / "test"
    pred_label_dir = pasta_predict / nome_experimento / "labels"

    img_files = sorted(pasta_teste.glob("*.png"))
    if not img_files:
        return
        
    samples = random.sample(img_files, min(10, len(img_files)))

    for img_path in samples:
        img = Image.open(img_path)
        w, h = img.size

        real_label = test_label_dir / (img_path.stem + ".txt")
        pred_label = pred_label_dir / (img_path.stem + ".txt")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, label_path, title, color in zip(axes, [real_label, pred_label], ["REAL", "PREDITO"], ["green", "red"]):
            ax.imshow(img)
            ax.set_title(f"{title} — {img_path.name}", fontsize=10)
            ax.axis("off")

            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            conf = parts[5] if len(parts) > 5 else None

                            x1 = (xc - bw / 2) * w
                            y1 = (yc - bh / 2) * h

                            rect = patches.Rectangle((x1, y1), bw * w, bh * h, linewidth=2, edgecolor=color, facecolor="none")
                            ax.add_patch(rect)

                            label = f"fire {conf}" if conf else "fire"
                            ax.text(x1, y1 - 6, label, color=color, fontsize=9, bbox=dict(facecolor="black", alpha=0.4, pad=1))

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    avaliar_modelo()