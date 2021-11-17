import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset


logger = logging.getLogger(__name__)

from utils import get_slot_labels


class InputExample(object):
    def __init__(self, guid, template_text, input_text, slot_labels=None):
        self.guid = guid
        self.template_text = template_text
        self.input_text = input_text
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                input_ids,
                input_attention_mask,
                input_token_type_ids,
                template_ids,
                template_attention_mask,
                template_token_type_ids,
                slot_labels_ids):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.input_token_type_ids = input_token_type_ids

        self.template_ids = template_ids
        self.template_attention_mask = template_attention_mask
        self.template_token_type_ids = template_token_type_ids

        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args

        self.data_file = 'data.json'

        self.slot_labels = get_slot_labels(args)

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, sample in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            input_text = sample['input'].split()
            # 2. template_text
            template_text = sample['template'].split()
            # 3. slot
            slot = sample['label']
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(input_text) == len(slot_labels)
            examples.append(InputExample(guid=guid, template_text=template_text, input_text=input_text, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(data=self._read_file(os.path.join(data_path, self.data_file)),
                                     set_type=mode)


processors = {
    "phobert": JointProcessor
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=0,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens_input = []
        slot_labels_ids = []
        for word, slot_label in zip(example.input_text, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens_input.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # Truncate
        if len(tokens_input) > max_seq_len - special_tokens_count:
            tokens_input = tokens_input[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens_input += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        input_token_type_ids = [0] * len(tokens_input)

        # Add [CLS] token
        tokens_input = [cls_token] + tokens_input
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        input_token_type_ids = [cls_token_segment_id] + input_token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens_input)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        input_attention_mask = input_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        input_token_type_ids = input_token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(input_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(input_attention_mask), max_seq_len)
        assert len(input_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(input_token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        tokens_template = []
        for word in example.template_text:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens_template.extend(word_tokens)

        if len(tokens_template) > max_seq_len - special_tokens_count:
            tokens_template = tokens_template[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens_template += [sep_token]
        template_token_type_ids = [0] * len(tokens_template)

        # Add [CLS] token
        tokens_template = [cls_token] + tokens_template
        template_token_type_ids = [cls_token_segment_id] + template_token_type_ids

        template_ids = tokenizer.convert_tokens_to_ids(tokens_template)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        template_attention_mask = [1 if mask_padding_with_zero else 0] * len(template_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(template_ids)
        template_ids = template_ids + ([pad_token_id] * padding_length)
        template_attention_mask = template_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        template_token_type_ids = template_token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(template_ids) == max_seq_len, "Error with input length {} vs {}".format(len(template_ids), max_seq_len)
        assert len(template_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(template_attention_mask), max_seq_len)
        assert len(template_token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(template_token_type_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input tokens: %s" % " ".join([str(x) for x in tokens_input]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_attention_mask: %s" % " ".join([str(x) for x in input_attention_mask]))
            logger.info("input_token_type_ids: %s" % " ".join([str(x) for x in input_token_type_ids]))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))
            logger.info("template tokens: %s" % " ".join([str(x) for x in tokens_template]))
            logger.info("template_ids: %s" % " ".join([str(x) for x in template_ids]))
            logger.info("template_attention_mask: %s" % " ".join([str(x) for x in template_attention_mask]))
            logger.info("template_token_type_ids: %s" % " ".join([str(x) for x in template_token_type_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_attention_mask=input_attention_mask,
                          input_token_type_ids=input_token_type_ids,
                          template_ids=template_ids,
                          template_attention_mask=template_attention_mask,
                          template_token_type_ids=template_token_type_ids,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.model_type](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}'.format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_attention_mask = torch.tensor([f.input_attention_mask for f in features], dtype=torch.long)
    all_input_token_type_ids = torch.tensor([f.input_token_type_ids for f in features], dtype=torch.long)
    
    all_template_ids = torch.tensor([f.template_ids for f in features], dtype=torch.long)
    all_template_attention_mask = torch.tensor([f.template_attention_mask for f in features], dtype=torch.long)
    all_template_token_type_ids = torch.tensor([f.template_token_type_ids for f in features], dtype=torch.long)

    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
                            all_input_attention_mask,
                            all_input_token_type_ids,
                            all_template_ids,
                            all_template_attention_mask,
                            all_template_token_type_ids,
                            all_slot_labels_ids)
    return dataset