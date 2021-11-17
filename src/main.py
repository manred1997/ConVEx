import argparse

import torch

from dataset import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, MODEL_CLASSES, MODEL_PATH_MAP

from transformers.trainer_utils import set_seed

def main(args):
    init_logger()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args)
    
    train_dataset = load_and_cache_examples(args, tokenizer, 'train')
    eval_dataset = load_and_cache_examples(args, tokenizer, 'dev')

    trainer = Trainer(args=args, train_dataset=train_dataset, test_dataset=eval_dataset)

    if args.do_train:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )

    parser.add_argument(
        "--model_type",
        default="phobert",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./HnBert", type=str, help="The pretrained model path")

    # Training Task
    parser.add_argument("--output_representation", type=int, default=128, help="Dimension of hidden representation after feed embedding representation of backbone")
    parser.add_argument("--num_attn_heads_enrich", type=int, default=4, help="Number of multi head attention of riched block")
    parser.add_argument("--hidden_inner", type=int, default=512, help="Dimension of inner hidden representation of FFN")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of enriched block")

    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")

    # Hyperparam training
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len", default=256, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    
    parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")
    
    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')


    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    main(args)