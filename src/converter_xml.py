import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["fire"] 

def converter_coordenadas(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_centro = (box[0] + box[1]) / 2.0
    y_centro = (box[2] + box[3]) / 2.0
    largura = box[1] - box[0]
    altura = box[3] - box[2]
    
    x = x_centro * dw
    w = largura * dw
    y = y_centro * dh
    h = altura * dh
    return (x, y, w, h)

def processar_pasta(caminho_base, split):
    # Onde estão os XMLs e onde vão ficar os TXTs (na pasta labels separada)
    pasta_xml = Path(caminho_base) / "labels" / split 
    pasta_txt = Path(caminho_base) / "labels" / split 
    
    if not pasta_xml.exists():
        print(f"[AVISO] Pasta não encontrada: {pasta_xml}")
        return

    pasta_txt.mkdir(parents=True, exist_ok=True)
    
    arquivos_xml = list(pasta_xml.glob("*.xml"))
    print(f"Convertendo {len(arquivos_xml)} arquivos em {split}...")
    
    # ... (o resto do código de conversão continua igual)

    if len(arquivos_xml) == 0:
        print(f"Nenhum arquivo .xml encontrado em {pasta_xml}. Verifique onde estão os XMLs originais.")
        return

    for arquivo_xml in arquivos_xml:
        tree = ET.parse(arquivo_xml)
        root = tree.getroot()
        
        tamanho = root.find('size')
        if tamanho is None:
            continue
        w = int(tamanho.find('width').text)
        h = int(tamanho.find('height').text)
        
        nome_arquivo_txt = arquivo_xml.stem + ".txt"
        caminho_saida = pasta_txt / nome_arquivo_txt
        
        with open(caminho_saida, 'w') as f_out:
            for obj in root.iter('object'):
                nome_classe = obj.find('name').text.lower()
                if nome_classe not in CLASSES:
                    continue
                    
                id_classe = CLASSES.index(nome_classe)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                bb = converter_coordenadas((w, h), b)
                f_out.write(f"{id_classe} {' '.join([f'{a:.6f}' for a in bb])}\n")

if __name__ == "__main__":
    # Caminho corrigido apontando para a pasta correta dentro de datasets/
    caminho_dataset = Path("datasets/flamevision-dataset/Detection")
    
    # Processa as 3 pastas (atente-se se o split de validação chama 'val' ou 'valid')
    processar_pasta(caminho_dataset, "train")
    processar_pasta(caminho_dataset, "val")   # Mude para "valid" se a pasta se chamar valid
    processar_pasta(caminho_dataset, "test")
    