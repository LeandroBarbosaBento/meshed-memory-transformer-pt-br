"""
Script para gerar legendas a partir de uma única imagem.

Fluxo:
  1. Carrega a imagem
  2. Extrai features visuais (grid 7x7 via ResNet-101 → tensor 49x2048)
  3. Passa as features pelo modelo Meshed-Memory Transformer
  4. Decodifica os tokens gerados em texto legível

Uso:
    python generate_caption.py --image caminho/para/imagem.jpg

Opções extras:
    --model_path    Caminho do checkpoint (.pth)   [default: saved_models/m2_transformer_best.pth]
    --vocab_path    Caminho do vocabulário (.pkl)   [default: vocab.pkl]
    --beam_size     Tamanho do beam search          [default: 5]
    --max_len       Comprimento máximo da legenda   [default: 20]
    --device        cuda ou cpu (auto-detecta)
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

from models.transformer import (
    Transformer,
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
)
from data import TextField


# ---------------------------------------------------------------------------
#  Extração de features
# ---------------------------------------------------------------------------
IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

GRID_TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    IMAGENET_NORMALIZE,
])


def build_resnet101_backbone(device):
    """Constrói backbone ResNet-101 (até layer4) para features 2048-d."""
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    backbone.eval().to(device)
    del resnet
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return backbone


def extract_features(image_path, backbone, device, max_detections=50):
    """
    Extrai grid features de uma imagem usando ResNet-101.
    Retorna tensor (1, max_detections, 2048) compatível com o modelo.
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = GRID_TRANSFORM(img).unsqueeze(0).to(device)  # (1, 3, 448, 448)

    adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)).to(device)

    with torch.no_grad():
        feat = backbone(img_tensor)       # (1, 2048, H', W')
        feat = adaptive_pool(feat)        # (1, 2048, 7, 7)
        feat = feat.view(1, 2048, 49)     # (1, 2048, 49)
        feat = feat.permute(0, 2, 1)      # (1, 49, 2048)

    # Ajusta para max_detections (padding com zeros se necessário)
    num_regions = feat.shape[1]  # 49
    if num_regions < max_detections:
        padding = torch.zeros(1, max_detections - num_regions, 2048, device=device)
        feat = torch.cat([feat, padding], dim=1)
    elif num_regions > max_detections:
        feat = feat[:, :max_detections, :]

    return feat  # (1, max_detections, 2048)


# ---------------------------------------------------------------------------
#  Geração de legenda
# ---------------------------------------------------------------------------
def generate_caption(model, features, text_field, beam_size=5, max_len=20):
    """Gera legenda via beam search e decodifica os tokens."""
    model.eval()
    with torch.no_grad():
        out, _ = model.beam_search(
            features,
            max_len,
            text_field.vocab.stoi['<eos>'],
            beam_size,
            out_size=1,
        )
    # Decodifica tokens → palavras
    caption = text_field.decode(out, join_words=False)
    # Remove duplicatas consecutivas (mesmo tratamento do test.py)
    caption = caption[0]  # pega a primeira (e única) do batch
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
        help="Caminho do arquivo de vocabulário (.pkl)",
    )
    parser.add_argument(
        "--beam_size", type=int, default=5,
        help="Tamanho do beam search (default: 5)",
    )
    parser.add_argument(
        "--max_len", type=int, default=20,
        help="Número máximo de tokens na legenda (default: 20)",
    )
    parser.add_argument(
        "--max_detections", type=int, default=50,
        help="Número de regiões visuais (default: 50, compatível com treino)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda' ou 'cpu'. Auto-detecta por padrão.",
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Usando device: {device}")

    # --- Vocabulário ---
    print(f"Carregando vocabulário de: {args.vocab_path}")
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True,
        tokenize='spacy', remove_punctuation=True, nopoints=False,
    )
    text_field.vocab = pickle.load(open(args.vocab_path, 'rb'))
    print(f"Vocabulário carregado ({len(text_field.vocab)} tokens)")

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

    # --- Extração de features ---
    print("Carregando backbone ResNet-101 para extração de features...")
    backbone = build_resnet101_backbone(device)
    print(f"Extraindo features da imagem: {args.image}")
    features = extract_features(args.image, backbone, device, args.max_detections)

    # Libera backbone (não é mais necessário)
    del backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Geração de legenda ---
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
