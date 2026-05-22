from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer
from bert_score import score as bert_score_fn

def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    # BERTScore usando BERTimbau (BERT treinado em portugues)
    keys = list(gts.keys())
    refs = [gts[k][0] for k in keys]
    hyps = [gen[k][0] for k in keys]
    _, _, F1 = bert_score_fn(hyps, refs, model_type="neuralmind/bert-large-portuguese-cased", num_layers=24, verbose=False)
    all_score['BERTScore'] = F1.mean().item()
    all_scores['BERTScore'] = F1.tolist()

    return all_score, all_scores
