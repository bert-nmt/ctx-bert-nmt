import pdb
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from . import MultiheadAttention


def squeeze_pad(x, mask):  # ? x B x _, B x ?
    # pdb.set_trace()
    b, t = mask.shape
    lens = (~mask).sum(-1)
    mask_ = torch.arange(lens.max()).to(lens.device)[None, :] >= lens[:, None]
    idx = torch.zeros_like(mask_).long()
    pos = torch.arange(t).to(mask.device).unsqueeze(0).repeat(b, 1)
    idx[~mask_] = pos[~mask]
    idx = idx.permute(1, 0).unsqueeze(-1).repeat(1, 1, x.shape[-1])
    x_ = x.gather(0, idx)
    return x_, mask_


class FlatHANAttention(nn.Module):

    def __init__(self, n_context, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, ctx_embedding=False):
        super().__init__()
        # pdb.set_trace()
        assert self_attention == False
        self.n_context = n_context
        self.attn = MultiheadAttention(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias,
                                       add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                       self_attention=self_attention,
                                       encoder_decoder_attention=encoder_decoder_attention)
        assert ctx_embedding is None

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=True, attn_mask=None):
        r'''
        query        : T1 x B x _
        key          : ? x B x _
        value        : ? x B x _
        key_padding_mask : B x (T2 * Nctx)
        '''
        # pdb.set_trace()
        return self.attn(query, key, value, key_padding_mask=key_padding_mask, incremental_state=incremental_state,
                         need_weights=need_weights, static_kv=static_kv, attn_mask=attn_mask)

    @staticmethod
    def prepare_attention_input(bert_encoder_out, bert_encoder_padding_mask, ctx_idx, ctx_mask):
        # pdb.set_trace()
        # select
        bert_encoder_out = bert_encoder_out[:, ctx_idx, :]  # T2 x B x _  ==>  T2 x Nctx x B x _
        bert_encoder_padding_mask = bert_encoder_padding_mask[ctx_idx, :]  # Nctx x B x T2
        # select segment
        bert_encoder_padding_mask[~ctx_mask] = 1
        # reshape
        b = ctx_mask.shape[1]
        bert_encoder_out = bert_encoder_out.reshape(-1, b, bert_encoder_out.shape[-1])  # (T2 * Nctx) x B x _
        bert_encoder_padding_mask = bert_encoder_padding_mask.permute(1, 2, 0).reshape(b, -1)  # B x (T2 * Nctx)
        # squeeze
        bert_encoder_out, bert_encoder_padding_mask = squeeze_pad(bert_encoder_out, bert_encoder_padding_mask)
        # => ? x B x _, B x ?
        return {
            'bert_encoder_out': bert_encoder_out,
            'bert_encoder_padding_mask': bert_encoder_padding_mask,
        }


class HierHANAttention(nn.Module):
    class SentAttention(MultiheadAttention):
        def reorder_incremental_state(self, incremental_state, new_order):
            # pdb.set_trace()
            input_buffer = self._get_input_buffer(incremental_state)
            if input_buffer is not None:
                for k, v in input_buffer.items():
                    v = v.reshape((self.n_context, -1) + v.shape[1:])
                    v = v.index_select(1, new_order).reshape((-1,) + v.shape[2:])
                    input_buffer[k] = v
                self._set_input_buffer(incremental_state, input_buffer)

    def __init__(self, n_context, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, ctx_embedding=False):
        super().__init__()
        # pdb.set_trace()
        self.n_context = n_context
        self.sent_attn = self.SentAttention(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias,
                                            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                            self_attention=self_attention,
                                            encoder_decoder_attention=encoder_decoder_attention)
        self.sent_attn.n_context = n_context
        self.doc_attn = MultiheadAttention(embed_dim, num_heads, kdim=embed_dim, vdim=embed_dim, dropout=dropout,
                                           bias=bias,
                                           add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                                           self_attention=self_attention,
                                           encoder_decoder_attention=encoder_decoder_attention)
        assert ctx_embedding is None

    def forward(self, query, key, value, ctx_mask, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        r'''
        query        : T1 x B x _
        key          : T2 x Nctx x B x _
        value        : T2 x Nctx x B x _
        mask         : Nctx x B
        key_padding_mask : Nctx x B x T2
        '''

        b = query.shape[1]
        n_context = self.n_context
        # pdb.set_trace()
        # incremental_state = None

        # sent-level
        sent_query = query.unsqueeze(1).repeat(1, n_context, 1, 1)  # T1 x B x _  ==>  T1 x Nctx x B x _
        sent_query = sent_query.reshape(query.shape[0], n_context * b, query.shape[-1])  # ==> T1 x (Nctx * B) x _
        sent_key = key.reshape(key.shape[0], n_context * b, key.shape[-1])  # T2 x Nctx x B x _  ==>  T2 x (Nctx * B) x _
        sent_value = value.reshape(value.shape[0], n_context * b, value.shape[-1])  # T2 x Nctx x B x _  ==>  T2 x (Nctx * B) x _
        sent_mask = key_padding_mask.reshape(-1, key_padding_mask.shape[-1])  # Nctx x B x T2  ==>  (Nctx * B) x T2
        # out
        out, _ = self.sent_attn(sent_query, sent_key, sent_value,
                                key_padding_mask=sent_mask,
                                incremental_state=incremental_state,
                                need_weights=need_weights, static_kv=static_kv,
                                attn_mask=attn_mask)  # T1 x (Nctx * B) x _

        # doc-level
        doc_query = query.reshape(1, -1, query.shape[-1])  # T1 x B x _  ==>  1 x (T1 * B) x _
        out = out.reshape(out.shape[0], n_context, b, out.shape[-1])  # T1 x (Nctx * B) x _  ==>  T1 x Nctx x B x _
        out = out.permute(1, 0, 2, 3).reshape(n_context, -1, out.shape[-1])  # ==>  Nctx x T1 x B x _  ==>  Nctx x (T1 * B) x _
        doc_mask = ctx_mask.transpose(0, 1).unsqueeze(0).repeat(query.shape[0], 1, 1)  # Nctx x B  ==>  T1 x B x Nctx
        doc_mask = doc_mask.reshape(-1, n_context)  # ==>  (T1 * B) x Nctx
        # squeeze
        # out
        final = self.doc_attn(doc_query, out, out,
                              key_padding_mask=1 - doc_mask,
                              incremental_state=None, need_weights=need_weights,
                              static_kv=False, attn_mask=attn_mask)  # final[0]: 1 x (T1 * B) x _
        return (final[0].reshape(-1, b, final[0].shape[-1]),) + final[1:]

    @staticmethod
    def prepare_attention_input(bert_encoder_out, bert_encoder_padding_mask, ctx_idx, ctx_mask):
        # pdb.set_trace()
        # select
        bert_encoder_out = bert_encoder_out[:, ctx_idx, :]  # T2 x B x _  ==>  T2 x Nctx x B x _
        bert_encoder_padding_mask = bert_encoder_padding_mask[ctx_idx, :]  # Nctx x B x T2
        return {
            'bert_encoder_out': bert_encoder_out,
            'bert_encoder_padding_mask': bert_encoder_padding_mask,
            'ctx_mask': ctx_mask,
        }


han_attention_modules = dict(
    flat=FlatHANAttention,
    hier=HierHANAttention,
)
han_attention_module_names = sorted(han_attention_modules.keys())


def get_han_attention_module(*args, attention_type=None, **kwargs):
    return han_attention_modules[attention_type](*args, **kwargs)


def prepare_attention_input(attention_type, out, padding_mask, ctx_idx, ctx_mask):
    return han_attention_modules[attention_type].prepare_attention_input(out, padding_mask, ctx_idx, ctx_mask)
