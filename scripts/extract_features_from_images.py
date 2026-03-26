"""
Script para extrair features das imagens do COCO para o Meshed-Memory Transformer.

Gera um arquivo HDF5 compatível com o ImageDetectionsField.
Cada imagem é armazenada com a chave `{image_id}_features` contendo um tensor (N, 2048).

Dois modos disponíveis:
  - grid:    Grid features 7x7 via ResNet-101 → (49, 2048) por imagem. Rápido.
  - regions: Faster R-CNN detecta regiões → crops passam pela ResNet-101 → (N, 2048).

Uso:
    conda activate m2features
    PYTHONUNBUFFERED=1 python scripts/extract_features_from_images.py \\
        --image_dirs coco/images/train2014 coco/images/val2014 coco/images/test2014 \\
        --output features/coco_detections.hdf5 \\
        --mode regions --max_detections 50

Requisitos (ambiente m2features):
    pip install torch torchvision h5py tqdm numpy pillow
"""

import argparse
import gc
import glob
import os
import re
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def get_image_paths(image_dirs):
    """Coleta todos os caminhos de imagens e seus IDs dos diretórios fornecidos."""
    image_paths = []
    image_ids = []
    for image_dir in image_dirs:
        paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
        )
        for p in paths:
            fname = os.path.basename(p)
            match = re.search(r"(\d{12})\.\w+$", fname)
            if match:
                img_id = int(match.group(1))
            else:
                nums = re.findall(r"\d+", fname)
                img_id = int(nums[-1]) if nums else None
            if img_id is not None:
                image_paths.append(p)
                image_ids.append(img_id)
    return image_paths, image_ids


def build_resnet101_backbone(device):
    """Constrói backbone ResNet-101 (até layer4) + AdaptiveAvgPool para features 2048-d."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    pool = nn.AdaptiveAvgPool2d((1, 1))
    backbone.eval().to(device)
    pool.to(device)
    # Libera o resnet completo da memória
    del resnet
    gc.collect()
    torch.cuda.empty_cache()
    return backbone, pool


def build_detector(device, score_threshold=0.2):
    """Constrói Faster R-CNN para detecção de regiões."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model


IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


# ---------------------------------------------------------------------------
#  Extração de features com grid (ResNet-101)
# ---------------------------------------------------------------------------
def extract_grid_features(args, device):
    """Extrai grid features usando ResNet-101. Cada imagem → (49, 2048)."""
    print("=" * 60)
    print("Modo: Grid Features (ResNet-101)")
    print("Cada imagem gera 49 regiões (grid 7x7) com 2048 dimensões")
    print("=" * 60)

    image_paths, image_ids = get_image_paths(args.image_dirs)
    print(f"Total de imagens encontradas: {len(image_paths)}")
    if len(image_paths) == 0:
        print("Nenhuma imagem encontrada! Verifique os diretórios.")
        return

    backbone, _ = build_resnet101_backbone(device)
    adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)).to(device)

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        IMAGENET_NORMALIZE,
    ])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed = 0
    skipped = 0

    with h5py.File(args.output, "a") as h5f:
        for idx in tqdm(range(len(image_paths)), desc="Extraindo grid features"):
            img_id = image_ids[idx]
            key = f"{img_id}_features"

            if key in h5f:
                skipped += 1
                continue

            img = Image.open(image_paths[idx]).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = backbone(img_tensor)          # (1, 2048, H', W')
                feat = adaptive_pool(feat)            # (1, 2048, 7, 7)
                feat = feat.view(1, 2048, 49)         # (1, 2048, 49)
                feat = feat.permute(0, 2, 1)          # (1, 49, 2048)

            h5f.create_dataset(key, data=feat[0].cpu().numpy().astype(np.float32))
            processed += 1

            # Flush periódico a cada 500 imagens
            if processed % 500 == 0:
                h5f.flush()
                print(f"  [checkpoint] {processed} processadas, {skipped} puladas")

        h5f.flush()

    print(f"\nConcluído! {processed} imagens processadas, {skipped} já existiam.")
    print(f"Arquivo salvo em: {args.output}")


# ---------------------------------------------------------------------------
#  Extração de features com regiões (Faster R-CNN + ResNet-101)
# ---------------------------------------------------------------------------
def extract_region_features(args, device):
    """
    Extrai region features:
      1. Faster R-CNN detecta bounding boxes de objetos
      2. Cada região é cropada, redimensionada e passada pela ResNet-101
      3. Features 2048-d por região são salvas no HDF5

    Otimizações vs. versão anterior:
      - Processa 1 imagem por vez no detector (evita OOM)
      - Crops são processados em mini-batches de 16 (evita VRAM explosion)
      - Flush do HDF5 a cada 200 imagens (evita corrupção)
      - workers=0 para reduzir uso de RAM
      - Limpeza explícita de CUDA cache
    """
    print("=" * 60)
    print("Modo: Region Features (Faster R-CNN + ResNet-101)")
    print(f"Até {args.max_detections} regiões por imagem com 2048 dimensões")
    print("=" * 60)

    image_paths, image_ids = get_image_paths(args.image_dirs)
    print(f"Total de imagens encontradas: {len(image_paths)}")
    if len(image_paths) == 0:
        print("Nenhuma imagem encontrada! Verifique os diretórios.")
        return

    # --- Carrega modelos ---
    print("Carregando Faster R-CNN...")
    detector = build_detector(device, args.score_threshold)
    print("Carregando ResNet-101 backbone...")
    backbone, pool = build_resnet101_backbone(device)
    print("Modelos carregados.\n")

    CROP_BATCH_SIZE = 16  # mini-batch para crops na ResNet
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    processed = 0
    skipped = 0
    t_start = time.time()

    with h5py.File(args.output, "a") as h5f:
        pbar = tqdm(range(len(image_paths)), desc="Extraindo region features")
        for idx in pbar:
            img_id = image_ids[idx]
            key = f"{img_id}_features"

            # Pula se já foi processada
            if key in h5f:
                skipped += 1
                continue

            # --- 1. Carrega imagem e detecta regiões ---
            img = Image.open(image_paths[idx]).convert("RGB")
            img_tensor = transforms.ToTensor()(img)  # (3, H, W), valores [0,1]

            with torch.no_grad():
                detections = detector([img_tensor.to(device)])

            det = detections[0]
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()

            # Filtra por score
            keep = scores >= args.score_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            # Limita detecções
            if len(scores) > args.max_detections:
                top_idx = np.argsort(scores)[::-1][:args.max_detections]
                boxes = boxes[top_idx]
                scores = scores[top_idx]

            # --- 2. Cropa regiões e extrai features via ResNet-101 ---
            _, H, W = img_tensor.shape

            if len(boxes) == 0:
                # Sem detecções: feature global da imagem inteira
                crop = transforms.Resize((224, 224))(img_tensor)
                crop = IMAGENET_NORMALIZE(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = pool(backbone(crop)).squeeze(-1).squeeze(-1)
                features = feat.cpu().numpy()
            else:
                # Prepara crops
                crops = []
                for box in boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img_tensor[:, y1:y2, x1:x2]
                    crop = transforms.Resize((224, 224))(crop)
                    crop = IMAGENET_NORMALIZE(crop)
                    crops.append(crop)

                if len(crops) == 0:
                    crop = transforms.Resize((224, 224))(img_tensor)
                    crop = IMAGENET_NORMALIZE(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = pool(backbone(crop)).squeeze(-1).squeeze(-1)
                    features = feat.cpu().numpy()
                else:
                    # Processa crops em mini-batches para não estourar VRAM
                    all_feats = []
                    for start in range(0, len(crops), CROP_BATCH_SIZE):
                        batch = torch.stack(crops[start:start + CROP_BATCH_SIZE]).to(device)
                        with torch.no_grad():
                            feat = pool(backbone(batch)).squeeze(-1).squeeze(-1)
                        all_feats.append(feat.cpu().numpy())
                        del batch, feat
                    features = np.concatenate(all_feats, axis=0)
                    del all_feats

            # --- 3. Salva no HDF5 ---
            h5f.create_dataset(key, data=features.astype(np.float32))
            processed += 1

            # Libera memória
            del img, img_tensor, detections, det, features
            if processed % 50 == 0:
                torch.cuda.empty_cache()

            # Flush periódico e status
            if processed % 200 == 0:
                h5f.flush()
                elapsed = time.time() - t_start
                rate = processed / elapsed
                remaining = (len(image_paths) - processed - skipped) / max(rate, 0.01)
                pbar.set_postfix(
                    saved=processed,
                    skipped=skipped,
                    rate=f"{rate:.1f} img/s",
                    eta=f"{remaining/3600:.1f}h",
                )

        # Flush final
        h5f.flush()

    elapsed = time.time() - t_start
    print(f"\nConcluído! {processed} imagens processadas, {skipped} já existiam.")
    print(f"Tempo total: {elapsed/3600:.1f} horas ({processed/max(elapsed,1):.1f} img/s)")
    print(f"Arquivo salvo em: {args.output}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extrai features das imagens COCO para treinar o Meshed-Memory Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Grid features (mais rápido, recomendado para começar):
  python scripts/extract_features_from_images.py \\
      --image_dirs coco/images/train2014 coco/images/val2014 \\
      --output features/coco_detections.hdf5 \\
      --mode grid

  # Region features (mais fiel ao paper original):
  python scripts/extract_features_from_images.py \\
      --image_dirs coco/images/train2014 coco/images/val2014 coco/images/test2014 \\
      --output features/coco_detections.hdf5 \\
      --mode regions --max_detections 50

  # Depois, para treinar:
  python train.py \\
      --features_path features/coco_detections.hdf5 \\
      --annotation_folder coco/annotations \\
      --exp_name m2_transformer
        """,
    )

    parser.add_argument(
        "--image_dirs", type=str, nargs="+",
        default=["coco/images/train2014", "coco/images/val2014", "coco/images/test2014"],
        help="Diretórios com as imagens COCO",
    )
    parser.add_argument(
        "--output", type=str, default="features/coco_detections.hdf5",
        help="Caminho do arquivo HDF5 de saída (default: features/coco_detections.hdf5)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["grid", "regions"], default="grid",
        help="'grid' = ResNet-101 grid 7x7 (rápido); 'regions' = Faster R-CNN + ResNet-101 (fiel ao paper)",
    )
    parser.add_argument(
        "--max_detections", type=int, default=50,
        help="Número máximo de regiões por imagem no modo 'regions' (default: 50)",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.2,
        help="Score mínimo para detecções no modo 'regions' (default: 0.2)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda ou cpu). Auto-detecta por padrão.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Usando device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    sys.stdout.flush()

    if args.mode == "grid":
        extract_grid_features(args, device)
    else:
        extract_region_features(args, device)

    print("\n✅ Extração concluída!")
    print(f"Para treinar o modelo, use:")
    print(f"  python train.py --features_path {args.output} --annotation_folder coco/annotations")


if __name__ == "__main__":
    main()
