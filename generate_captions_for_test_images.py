import gc
import os
import pickle

import torch

from generate_caption import (
    build_detector,
    build_resnet101_backbone,
    extract_features,
    generate_caption,
)
from models.transformer import (
    MemoryAugmentedEncoder,
    MeshedDecoder,
    ScaledDotProductAttentionMemory,
    Transformer,
)
from data import TextField

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
IMG_DIR = "img_testes"
MODEL_PATH = "saved_models/m2_transformer_v5_best.pth"
VOCAB_PATH = "vocab_m2_transformer_v5.pkl"
BEAM_SIZE = 5
MAX_LEN = 20
MAX_DETECTIONS = 50
SCORE_THRESHOLD = 0.2

images = [
    "cachorro_mulher.jpg",
    "correndo.jpg",
    "crianca_garrafa_bebendo.jpg",
    "crianca_pipa.jpg",
    "crianca.png",
    "estacionamento.png",
    "gato_natal.jpg",
    "homem_baseball.jpeg",
    "homem_cavalo_cachorro.jpg",
    "homem_cavalo.jpg",
    "homem_computador.jpg",
    "homem_faca_cozinha.jpg",
    "homem_sala.jpg",
    "menino_patinete.jpg",
    "montanha.png",
    "mulher_escalando.jpg",
    "mulher_geladeira.jpg",
    "mulher_lendo.jpg",
    "mulher_mochila.jpg",
    "mulher_telefone.jpg",
    "mulheres_anotando.jpg",
    "mulheres_baseball.jpg",
    "pessoa_aviao.jpg",
    "pessoas_cama.jpg",
    "senhor.png",
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # --- Vocabulario ---
    print(f"Carregando vocabulario de: {VOCAB_PATH}")
    text_field = TextField(
        init_token="<bos>", eos_token="<eos>", lower=True,
        tokenize="spacy", remove_punctuation=True, nopoints=False,
    )
    text_field.vocab = pickle.load(open(VOCAB_PATH, "rb"))
    print(f"Vocabulario carregado ({len(text_field.vocab)} tokens)")

    # --- Modelo ---
    print(f"Carregando modelo de: {MODEL_PATH}")
    encoder = MemoryAugmentedEncoder(
        3, 0,
        attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={"m": 40},
    )
    decoder = MeshedDecoder(
        len(text_field.vocab), 62, 3,
        text_field.vocab.stoi["<pad>"],
    )
    model = Transformer(text_field.vocab.stoi["<bos>"], encoder, decoder).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("Modelo carregado com sucesso!")

    # --- Detectores / backbone (carregados uma unica vez) ---
    print("Carregando Faster R-CNN e backbone ResNet-101...")
    detector = build_detector(device)
    backbone, pool = build_resnet101_backbone(device)

    # --- Loop sobre as imagens ---
    print("\n" + "=" * 60)
    results = {}
    for img_name in images:
        img_path = os.path.join(IMG_DIR, img_name)
        print(f"\nProcessando: {img_path}")
        try:
            features = extract_features(
                img_path, backbone, pool, detector, device,
                max_detections=MAX_DETECTIONS,
                score_threshold=SCORE_THRESHOLD,
            )
            caption = generate_caption(
                model, features, text_field,
                beam_size=BEAM_SIZE,
                max_len=MAX_LEN,
            )
        except Exception as e:
            caption = f"ERRO: {e}"

        results[img_name] = caption
        print(f"  Legenda: {caption}")

    # --- Sumario final ---
    print("\n" + "=" * 60)
    print("SUMARIO DAS LEGENDAS GERADAS")
    print("=" * 60)
    for img_name, caption in results.items():
        print(f"{img_name:40s} -> {caption}")
    print("=" * 60)

    # --- Libera recursos ---
    del detector, backbone, pool
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
