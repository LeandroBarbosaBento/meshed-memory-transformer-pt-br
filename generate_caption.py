"""
Script para gerar legendas a partir de uma unica imagem.

Fluxo:
  1. Carrega a imagem
  2. Extrai features visuais (Faster R-CNN + ResNet-101) - mesmo pipeline do treino
  3. Passa as features pelo modelo Meshed-Memory Transformer
  4. Decodifica os tokens via beam search (igual ao test.py)

Uso:
    python generate_caption.py --image caminho/para/imagem.jpg \
        --model_path saved_models/m2_transformer_v4_best.pth \
        --vocab_path vocab_m2_transformer_v4.pkl
"""

import argparse
import gc
import itertools
import pickle

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from models.transformer import (
    Transformer,
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
)
from data import TextField


# ---------------------------------------------------------------------------
#  Extracao de features (mesmo pipeline do extract_features_from_images.py)
# ---------------------------------------------------------------------------
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def build_resnet101_backbone(device):
    """Constroi backbone ResNet-101 (ate layer4) + AdaptiveAvgPool para features 2048-d."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    pool = nn.AdaptiveAvgPool2d((1, 1))
    backbone.eval().to(device)
    pool.to(device)
    del resnet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return backbone, pool


def build_detector(device):
    """Constroi Faster R-CNN para deteccao de regioes."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model


def extract_features(image_path, backbone, pool, detector, device,
                     max_detections=50, score_threshold=0.01):
    """
    Extrai region features de uma imagem - mesmo pipeline usado para gerar
    o HDF5 de treino (extract_features_from_images.py modo regions).

    1. Faster R-CNN detecta bounding boxes
    2. Cada regiao e cropada, redimensionada (224x224) e passa pela ResNet-101
    3. Retorna tensor (1, max_detections, 2048) com padding se necessario
    """
    CROP_BATCH_SIZE = 16

    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img)  # (3, H, W), valores [0,1]

    # --- 1. Detecta regioes ---
    with torch.no_grad():
        detections = detector([img_tensor.to(device)])

    det = detections[0]
    boxes = det["boxes"].cpu().numpy()
    scores = det["scores"].cpu().numpy()

    # Filtra por score
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]

    # Limita deteccoes (top-k por score)
    if len(scores) > max_detections:
        top_idx = np.argsort(scores)[::-1][:max_detections]
        boxes = boxes[top_idx]

    # --- 2. Cropa regioes e extrai features via ResNet-101 ---
    _, H, W = img_tensor.shape

    if len(boxes) == 0:
        # Sem deteccoes: feature global da imagem inteira
        crop = transforms.Resize((224, 224))(img_tensor)
        crop = IMAGENET_NORMALIZE(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            features = pool(backbone(crop)).squeeze(-1).squeeze(-1)
        features = features.cpu().numpy()
    else:
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
                features = pool(backbone(crop)).squeeze(-1).squeeze(-1)
            features = features.cpu().numpy()
        else:
            all_feats = []
            for start in range(0, len(crops), CROP_BATCH_SIZE):
                batch = torch.stack(crops[start:start + CROP_BATCH_SIZE]).to(device)
                with torch.no_grad():
                    feat = pool(backbone(batch)).squeeze(-1).squeeze(-1)
                all_feats.append(feat.cpu().numpy())
                del batch, feat
            features = np.concatenate(all_feats, axis=0)

    # --- 3. Padding/truncamento para (max_detections, 2048) ---
    if features.ndim == 1:
        features = features.reshape(1, -1)

    delta = max_detections - features.shape[0]
    if delta > 0:
        features = np.concatenate([features, np.zeros((delta, features.shape[1]))], axis=0)
    elif delta < 0:
        features = features[:max_detections]

    return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
#  Geracao de legenda (igual ao test.py)
# ---------------------------------------------------------------------------
def generate_caption(model, features, text_field, beam_size=5, max_len=20):
    """
    Gera legenda via beam search e decodifica os tokens.
    Mesmo fluxo de predict_captions() do test.py.
    """
    model.eval()
    with torch.no_grad():
        out, _ = model.beam_search(
            features,
            max_len,
            text_field.vocab.stoi['<eos>'],
            beam_size,
            out_size=1,
        )
    # Decodifica tokens -> palavras (igual test.py)
    caps_gen = text_field.decode(out, join_words=False)
    caption = caps_gen[0]
    caption = ' '.join([k for k, g in itertools.groupby(caption)])
    return caption.strip()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gera legenda para uma imagem usando o Meshed-Memory Transformer",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Caminho da imagem de entrada (jpg, png, etc.)",
    )
    parser.add_argument(
        "--model_path", type=str, default="saved_models/m2_transformer_best.pth",
        help="Caminho do checkpoint do modelo (.pth)",
    )
    parser.add_argument(
        "--vocab_path", type=str, default="vocab_m2_transformer.pkl",
        help="Caminho do arquivo de vocabulario (.pkl)",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Tamanho do beam search (default: 5)",
    )
    parser.add_argument(
        "--max_len", type=int, default=20,
        help="Numero maximo de tokens na legenda (default: 20)",
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.01,
        help="Score minimo do Faster R-CNN para deteccoes (default: 0.01)",
    )
    parser.add_argument(
        "--max_detections", type=int, default=50,
        help="Numero maximo de regioes visuais (default: 50)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' ou 'cpu'. Auto-detecta por padrao.",
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Usando device: {device}")

    # --- Vocabulario ---
    print(f"Carregando vocabulario de: {args.vocab_path}")
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True,
        tokenize='spacy', remove_punctuation=True, nopoints=False,
    )
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))
    print(f"Vocabulario carregado ({len(text_field.vocab)} tokens)")

    # --- Modelo ---
    print(f"Carregando modelo de: {args.model_path}")
    encoder = MemoryAugmentedEncoder(
        3, 0,
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={'m': 40},
    )
    decoder = MeshedDecoder(
        len(text_field.vocab), 62, 3,
        text_field.vocab.stoi['<pad>'],
    )
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Modelo carregado com sucesso!")

    # --- Extracao de features (Faster R-CNN + ResNet-101) ---
    print("Carregando Faster R-CNN...")
    detector = build_detector(device)
    print("Carregando backbone ResNet-101...")
    backbone, pool = build_resnet101_backbone(device)
    print(f"Extraindo features da imagem: {args.image}")
    features = extract_features(
        args.image, backbone, pool, detector, device,
        max_detections=args.max_detections,
        score_threshold=args.score_threshold,
    )

    # Libera modelos de extracao
    del detector, backbone, pool
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Geracao de legenda ---
    print("Gerando legenda...")
    caption = generate_caption(
        model, features, text_field,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )

    print("\n" + "=" * 60)
    print(f"Imagem:  {args.image}")
    print(f"Legenda: {caption}")
    print("=" * 60)


if __name__ == "__main__":
    main()
