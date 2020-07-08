#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig

from .decotor import BiGRU


class SoftMaskedBert(nn.Module):
    """
    Soft-Masked Bert
    论文：https://arxiv.org/pdf/2005.07421.pdf
    """
    def __init__(self, bert, tokenizer, hidden, layer_n, device):
        super(SoftMaskedBert, self).__init__()
        # self.bert = bert
        # self.tokenizer = tokenizer
        self.embedding = bert.embeddings.to(device)
        self.config = bert.config
        embedding_size = self.config.to_dict()['hidden_size']

        self.detector = BiGRU(embedding_size, hidden, layer_n)
        self.corrector = bert.encoder
        mask_token_id = torch.tensor([[tokenizer.mask_token_id]]).to(device)
        self.mask_e = self.embedding(mask_token_id)
        self.linear = nn.Linear(embedding_size, self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, input_mask, segment_ids):
        e = self.embedding(input_ids=input_ids, token_type_ids=segment_ids)
        p = self.detector(e)
        e_ = p * self.mask_e + (1-p) * e
        _, _, _, _, \
        _, \
        head_mask, \
        encoder_hidden_states, \
        encoder_extended_attention_mask= self._init_inputs(input_ids, input_mask)
        h = self.corrector(e_,
                           attention_mask=encoder_extended_attention_mask,
                           head_mask=head_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_extended_attention_mask)
        h = h[0] + e
        out = self.softmax(self.linear(h))
        return out, p

    def _init_inputs(self,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return input_ids, position_ids, token_type_ids, inputs_embeds, \
               extended_attention_mask, head_mask, encoder_hidden_states, encoder_extended_attention_mask


# if __name__ == "__main__":
#     config = BertConfig.from_pretrained('../data/chinese_wwm_pytorch/bert_config.json')
#     tokenizer = BertTokenizer.from_pretrained('../data/chinese_wwm_pytorch/vocab.txt')
#     bert = BertModel.from_pretrained('../data/chinese_wwm_pytorch/pytorch_model.bin', config=config)
#     model = SoftMaskedBert(bert, tokenizer, 2, 1)
#     text = '中国的'
#     token = tokenizer.tokenize(text)
#     ids = tokenizer.convert_tokens_to_ids(token)
#     ids = torch.Tensor([ids]).long()
#     print(ids)
#     out = model(ids)
#     # out = bert(ids)
#     print(out)