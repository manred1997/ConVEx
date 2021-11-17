from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from model import EnrichBlock, Feedforward, SlotClassifier

import torch.nn as nn
from torchcrf import CRF


class ConVEx(RobertaPreTrainedModel):
    def __init__(self, config, args):
        super(ConVEx, self).__init__(config)

        self.args = args
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.ffn_input = Feedforward(config, args)
        self.ffn_temp = Feedforward(config, args)
        self.layer_stack = nn.ModuleList([
            EnrichBlock(n_head=config.num_attention_heads,
                                        d_k=args.output_representation//args.num_attn_heads_enrich,
                                        d_v=args.output_representation//args.num_attn_heads_enrich,
                                        d_model=args.output_representation,
                                        d_inner=args.hidden_inner,
                                        dropout=args.dropout)
            for _ in range(args.n_layers)
        ])
        self.slot_classifier = SlotClassifier(args.output_representation, 4, args.droupout)

        if args.use_crf:
            self.crf = CRF(num_tags=4, batch_first=True)

    def forward(self,
                input_ids,
                input_attn_mask,
                template_ids,
                template_attn_mask,
                trg_seq=None,
                return_attns=False):
        
        enc_input = self.roberta(input_ids=input_ids,
                                attention_mask=input_attn_mask)[0]
        enc_input = self.ffn_input(enc_input)

        enc_template = self.roberta(input_ids=template_ids,
                                attention_mask=template_attn_mask)[0]
        enc_template = self.ffn_temp(enc_template)

        input_slf_attn_list, input_temp_attn_list = [], []

        for enrich_block in self.layer_stack:
            enc_input, input_slf_attn, input_temp_attn = enrich_block(
                enc_input, enc_template, slf_attn_mask=input_attn_mask, enc_temp_attn_mask=template_attn_mask)
            input_slf_attn_list += [input_slf_attn] if return_attns else []
            input_temp_attn_list += [input_temp_attn] if return_attns else []

        slot_logits = self.slot_classifier(enc_input)

        # Loss CRF
        total_loss = 0
        if trg_seq is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, trg_seq, mask=input_attn_mask.byte(), reduction='mean')
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if input_attn_mask is not None:
                    active_loss = input_attn_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, 4)[active_loss]
                    active_labels = trg_seq.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, 4), trg_seq.view(-1))
                
            total_loss += slot_loss
        
        outputs = ((slot_logits),)  # add hidden states and attention if they are here
        outputs = (total_loss,) + outputs
        
        if return_attns:
            return outputs, (input_slf_attn_list, input_temp_attn_list)
        return outputs
