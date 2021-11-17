import logging

from transformers import (
    AutoTokenizer,
    RobertaConfig
)

from model import ConVEx

import os
from seqeval.metrics import f1_score, precision_score, recall_score

MODEL_CLASSES = {
    "phobert": (RobertaConfig, ConVEx, AutoTokenizer)
}

MODEL_PATH_MAP = {
    "phobert" : "vinai/phobert-base"
}

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def load_tokenizer(args):
    # return MODEL_CLASSES[args.model_type][2].from_pretrained("/workspace/vinbrain/minhnp/pretrainedLM/phobert-base")
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)

def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, 'label.txt'), 'r', encoding='utf-8')]

def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    
    results.update(slot_result)
    return results
    
def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }