import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO DE DIRETÓRIOS LOCAIS
# ============================================================
BASE_DIR = Path(".")

PASTA_ORIGEM = BASE_DIR / "datasets" / "flamevision-dataset" 

# Pasta de saída (onde o dataset pronto para o YOLO vai ficar)
OUT = BASE_DIR / "datasets" / "flamevision-dataset"

print("Iniciando preparação local do dataset...")
print(f"Lendo imagens de : {PASTA_ORIGEM}")
print(f"Pasta de saída   : {OUT}")

if not PASTA_ORIGEM.exists():
    print(f"\nERRO: A pasta {PASTA_ORIGEM} não foi encontrada!")
    print("Verifique se você descompactou o dataset nesse local.")
    exit(1)

# ============================================================
# 1. LOCALIZANDO A PASTA "Detection"
# ============================================================
# O script procura automaticamente a subpasta "Detection" lá dentro
candidatos = [p for p in PASTA_ORIGEM.rglob("*") if p.is_dir() and p.name == "Detection"]

if not candidatos:
    print("\nERRO: Pasta 'Detection' não encontrada dentro da pasta de origem.")
    exit(1)

BASE = candidatos[0]
print(f"\nBase do subconjunto definida: {BASE}")

# ============================================================
# 2. PADRONIZAÇÃO DO DATASET PARA O FORMATO YOLO
# ============================================================
def voc_to_yolo(xml_path):
    """
    Converte uma anotação Pascal VOC (XML) para o formato YOLO (TXT).
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    size_tag = root.find("size")
    if size_tag is None:
        return []

    w = int(size_tag.find("width").text)
    h = int(size_tag.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue

        xmin_tag = bb.find("xmin")
        ymin_tag = bb.find("ymin")
        xmax_tag = bb.find("xmax")
        ymax_tag = bb.find("ymax")

        if None in (xmin_tag, ymin_tag, xmax_tag, ymax_tag):
            continue

        xmin = float(xmin_tag.text)
        ymin = float(ymin_tag.text)
        xmax = float(xmax_tag.text)
        ymax = float(ymax_tag.text)

        # Ajusta coordenadas aos limites da imagem
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)

        # Descarta boxes inválidas
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2) / w
        cy = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return yolo_lines

# Mapeamento dos splits
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

def converter_split(split_origem, split_destino):
    ann_dir = BASE / split_origem / "annotations"
    img_dir = BASE / split_origem / "images"
    img_out = OUT / "images" / split_destino
    lbl_out = OUT / "labels" / split_destino
    convertidas, ignoradas = 0, 0

    for xml_path in sorted(ann_dir.glob("*.xml")):
        img_candidates = list(img_dir.glob(f"{xml_path.stem}.*"))

        if not img_candidates:
            ignoradas += 1
            continue

        img_path = img_candidates[0]
        yolo_lines = voc_to_yolo(xml_path)

        if not yolo_lines:
            ignoradas += 1
            continue

        # Copia imagem
        shutil.copy2(img_path, img_out / img_path.name)
        # Cria arquivo .txt com labels YOLO
        (lbl_out / f"{xml_path.stem}.txt").write_text("\n".join(yolo_lines))
        convertidas += 1

    return convertidas, ignoradas

# Limpa diretório de saída se já existir para não misturar arquivos antigos
if OUT.exists():
    shutil.rmtree(OUT)

# Cria estrutura de pastas do YOLO
for split in ["train", "val", "test"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

print("\nIniciando conversão das anotações de XML para YOLO (TXT)...")
for split_origem, split_destino in SPLIT_MAP.items():
    convertidas, ignoradas = converter_split(split_origem, split_destino)
    print(f"Pasta '{split_destino:5s}' concluída | convertidas: {convertidas:5d} | ignoradas: {ignoradas:5d}")

# ============================================================
# 3. GERAÇÃO DO ARQUIVO DATASET.YAML
# ============================================================
yaml_content = f"""\
path: {OUT.resolve().as_posix()}

train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['fire']
"""

(OUT / "dataset.yaml").write_text(yaml_content)
print("\nArquivo dataset.yaml gerado com sucesso!")

# ============================================================
# 4. VALIDAÇÃO FINAL
# ============================================================
print("\nValidação Final:")
EXTENSOES_IMG = (".jpg", ".jpeg", ".png")

for split in ["train", "val", "test"]:
    n_img = sum(1 for p in (OUT / "images" / split).iterdir() if p.suffix.lower() in EXTENSOES_IMG)
    n_lbl = len(list((OUT / "labels" / split).glob("*.txt")))
    print(f"  {split:6s}: imagens={n_img:4d} | labels={n_lbl:4d}")

print(f"\n✅ PREPARAÇÃO CONCLUÍDA! O dataset está pronto para o YOLO na pasta: {OUT}")