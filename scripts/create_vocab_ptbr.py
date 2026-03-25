#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para criar vocabulário em português a partir das legendas COCO em PT-BR
"""

import json
import pickle
import argparse
from collections import Counter
from tqdm import tqdm


class Vocabulary:
    """Classe para criar e gerenciar o vocabulário"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        """Adiciona uma palavra ao vocabulário"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        """Retorna o índice de uma palavra"""
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        """Retorna o tamanho do vocabulário"""
        return len(self.word2idx)


def build_vocab(json_file, threshold=5):
    """
    Constrói vocabulário a partir das legendas
    
    Args:
        json_file: caminho para o arquivo JSON com legendas
        threshold: frequência mínima para incluir palavra no vocab
    
    Returns:
        vocab: objeto Vocabulary
    """
    print(f"\n📖 Carregando legendas de: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    print(f"✅ Carregadas {len(annotations)} legendas")
    
    # Contar frequência das palavras
    print("\n🔢 Contando frequência das palavras...")
    counter = Counter()
    for annotation in tqdm(annotations, desc="Processando"):
        caption = annotation['caption']
        # Tokenização simples (split por espaços)
        tokens = caption.lower().split()
        counter.update(tokens)
    
    print(f"\n📊 Total de palavras únicas: {len(counter)}")
    
    # Filtrar palavras por frequência
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    print(f"📊 Palavras com frequência >= {threshold}: {len(words)}")
    
    # Criar vocabulário
    print("\n🏗️  Construindo vocabulário...")
    vocab = Vocabulary()
    
    # Adicionar tokens especiais
    vocab.add_word('<pad>')   # padding
    vocab.add_word('<start>') # início de sequência
    vocab.add_word('<end>')   # fim de sequência
    vocab.add_word('<unk>')   # palavra desconhecida
    
    # Adicionar palavras filtradas
    for word in tqdm(words, desc="Adicionando palavras"):
        vocab.add_word(word)
    
    return vocab


def main():
    parser = argparse.ArgumentParser(description='Criar vocabulário em português')
    parser.add_argument('--json_file', type=str, 
                        default='coco/annotations/captions_train2014.json',
                        help='Arquivo JSON com legendas de treino')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Frequência mínima para incluir palavra')
    parser.add_argument('--output', type=str, default='vocab_ptbr.pkl',
                        help='Nome do arquivo de saída')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🇧🇷  CRIAÇÃO DE VOCABULÁRIO EM PORTUGUÊS")
    print("=" * 70)
    
    # Construir vocabulário
    vocab = build_vocab(args.json_file, args.threshold)
    
    # Salvar vocabulário
    print(f"\n💾 Salvando vocabulário em: {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"\n✅ Vocabulário criado com sucesso!")
    print(f"   📊 Tamanho total: {len(vocab)} palavras")
    print(f"   📁 Arquivo: {args.output}")
    
    # Mostrar exemplos
    print("\n📝 Exemplos de palavras no vocabulário:")
    sample_words = list(vocab.word2idx.keys())[:20]
    for i, word in enumerate(sample_words, 1):
        idx = vocab.word2idx[word]
        print(f"   {i:2d}. '{word}' -> {idx}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
