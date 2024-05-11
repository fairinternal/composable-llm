"""
Evaluation metric

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import torch


def attention_metrics(sequence, attentions):
    """
    Compute success metrics to CoT emergence.

    Parameters
    ----------
    sequence: tensor of size (bsz, seq_len)
        Token sequences.
    attentions: tensore of size (n_layer=2, bsz, n_head=1, seq_len, seq_len)
        Attention maps.

    Returns
    -------
    attn_inv: tensor of size (len, n_layer * n_head = 2)
        Score of invarance of attention to token sequence. Lower is better.
    attn_peaky: tensore of size (len, 2 * n_layer * n_head = 4)
        Success metrics for the attention maps. Higher is better.
    """
    eois = torch.argmax((sequence == 1).to(int), dim=1)
    all_eois = torch.unique(eois)

    attn_inv = torch.empty((len(all_eois), 2), device=eois.device, dtype=float)
    attn_peaky = torch.empty((len(all_eois), 4), device=eois.device, dtype=float)

    # group and process sequences by lengths
    for i, eoi in enumerate(all_eois):
        ind = eois == eoi

        # handcrafted EoS given EoI
        eos = 2 * eoi

        # handcrafted attention score to look at
        attn0 = attentions[0, ind, 0, eoi + 1 : eos, eoi]
        attn1 = torch.diagonal(attentions[1, ind, 0, eoi : eos - 1, 1:eoi], dim1=1, dim2=2)

        # how does attention change for different sequences
        attn_inv[i, 0] = attn0.std(dim=0).mean()
        attn_inv[i, 1] = attn1.std(dim=0).mean()

        # how much the attention is picky
        attn_peaky[i, 0] = attn0.mean()
        attn_peaky[i, 1] = (attn0 > 0.5).to(dtype=float).mean()
        attn_peaky[i, 2] = attn1.mean()
        attn_peaky[i, 3] = (attn1 > 0.5).to(dtype=float).mean()
    return attn_inv, attn_peaky
