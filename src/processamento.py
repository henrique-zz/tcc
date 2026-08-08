import cv2
import numpy as np
import shutil
import random
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

def _expand_box(box, margin):
    x1, y1, x2, y2 = box
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)

def _intersects(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def merge_boxes(boxes, margin):
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            merged = True
            while merged:
                merged = False
                for j in range(len(boxes)):
                    if i == j or used[j]:
                        continue
                    b1 = _expand_box((x1, y1, x2, y2), margin)
                    b2 = _expand_box(boxes[j], margin)
                    if _intersects(b1, b2):
                        bx1, by1, bx2, by2 = boxes[j]
                        x1, y1 = min(x1, bx1), min(y1, by1)
                        x2, y2 = max(x2, bx2), max(y2, by2)
                        used[j] = True
                        merged = True
                        changed = True
            used[i] = True
            new_boxes.append((x1, y1, x2, y2))
        boxes = new_boxes
    return boxes

def _clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))

def _is_valid_box(box):
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1

def preparar_dataset(pasta_origem: Path, pasta_destino: Path, config: dict):
    """
    Função genérica que lê máscaras (_gt.png) de uma pasta de origem,
    processa as bounding boxes e salva no formato YOLO na pasta de destino.
    """
    kernel_size = config['kernel_size']
    merge_margin = config['merge_margin']
    min_area_pct = config['min_area_pct']
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 1. CLEAN OUTPUT
    if pasta_destino.exists():
        shutil.rmtree(pasta_destino)
    for split in ["train", "val", "test"]:
        (pasta_destino / "images" / split).mkdir(parents=True, exist_ok=True)
        (pasta_destino / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. FIND FILES & GENERATE LABELS
    gt_files = sorted(pasta_origem.rglob("*_gt.png"))
    dataset = []
    
    for gt_path in gt_files:
        rgb_path = Path(str(gt_path).replace("_gt.png", "_rgb.png"))
        if not rgb_path.exists():
            continue

        mask = np.array(Image.open(gt_path))
        h, w = mask.shape[:2]
        
        mask = (mask > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        min_area = h * w * min_area_pct
        boxes = []
        
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area >= min_area:
                boxes.append((x, y, x + bw, y + bh))
                
        boxes = merge_boxes(boxes, margin=merge_margin)
        boxes = [_clip_box(b, w, h) for b in boxes]
        boxes = [b for b in boxes if _is_valid_box(b)]
        
        if boxes:
            dataset.append((rgb_path, boxes))

    # 3. SPLIT
    train_data, temp = train_test_split(dataset, test_size=(1.0 - config['train_size']), random_state=42)
    val_pct = config['val_size'] / (config['val_size'] + config['test_size'])
    val_data, test_data = train_test_split(temp, test_size=(1.0 - val_pct), random_state=42)

    splits = {"train": train_data, "val": val_data, "test": test_data}

    # 4. SAVE LABELS & YAML
    for split_name, split_data in splits.items():
        for rgb_path, boxes in split_data:
            safe_name = f"{rgb_path.parent.name}_{rgb_path.name}"
            img_out = pasta_destino / "images" / split_name / safe_name
            label_out = pasta_destino / "labels" / split_name / (Path(safe_name).stem + ".txt")
            
            shutil.copy2(rgb_path, img_out)
            img_w, img_h = Image.open(rgb_path).size
            
            lines = []
            for x1, y1, x2, y2 in boxes:
                xc = ((x1 + x2) / 2) / img_w
                yc = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                
            with open(label_out, "w") as f:
                f.write("\n".join(lines))

    yaml_text = f"""\
path: {pasta_destino.resolve().as_posix()}
train: images/train
val:   images/val
test:  images/test
names:
  0: fire
"""
    with open(pasta_destino / "dataset.yaml", "w") as f:
        f.write(yaml_text)
        
    print(f"Dataset '{pasta_destino.name}' preparado! Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")