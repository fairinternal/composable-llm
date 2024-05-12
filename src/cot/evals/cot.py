"""
Evaluation metric

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""

import torch


def attention_metrics(sequences, attentions):
    """
    Compute success metrics to CoT emergence.

    Parameters
    ----------
    sequences: tensor of size (bsz, seq_len)
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
    eois = torch.argmax((sequences == 1).to(int), dim=1)
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


class SimpleEval:
    def __init__(self, lengths):
        self.eval_dim = 2 * len(lengths) + 6

    def __call__(self, model, trainset, testset):
        _, seq_err, spe_err = self.eval_model(trainset, model, special=True)
        _, test_seq_err, test_spe_err = self.eval_model(testset, model, special=True)
        return torch.concat((1 - seq_err.cpu(), 1 - test_seq_err.cpu(), 1 - spe_err.cpu(), 1 - test_spe_err.cpu()))

    def eval_model(self, dataset, model, batch_size=None, special=False):
        """
        Eval model on dataset

        Parameters
        ----------
        dataset: SequenceDataset
            dataset to evaluate the model on.
        model: torch.nn.Module
            model to be evaluated.
        batch_size: int, optional
            batch size to use when data does not fit in memory.
        special: bool, optional (default is False)
            whether to compute special token syntaxic error.

        Returns
        -------
        err_by_len: torch.Tensor
            token errors average by lengths.
        seq_err_by_len: torch.Tensor
            sequence errors average by lengths.
        spe_err: torch.Tensor of size (-1, 3)
            sequence of special token erros.
        """

        if batch_size is None:
            batch_size = len(dataset.data)

        device = list(model.parameters())[0].device

        nb_data = len(dataset.data)
        err = torch.empty(nb_data, device=device, dtype=float)
        seq_err = torch.empty(nb_data, device=device, dtype=bool)
        if special:
            spe_err = torch.zeros(3, device=device, dtype=float)

        begin = 0
        for end in range(batch_size, nb_data + batch_size, batch_size):
            data = dataset.data[begin:end].to(device=device, dtype=torch.long)
            pred = model(data[:, :-1]).argmax(dim=-1)
            ground_truth = data[:, 1:]

            ind = ground_truth == 1
            cot_mask = ind.cumsum(axis=1)
            cot_mask[ind] = 0
            cot_mask = cot_mask.to(dtype=bool)
            pred[~cot_mask] = ground_truth[~cot_mask]

            errors = pred != ground_truth
            seq_err[begin:end] = errors.any(dim=1)
            err[begin:end] = errors.float().mean(dim=1)
            if special:
                tmp = self.eval_spe_tok_err(pred)
                spe_err[:] += torch.stack(tmp) * (end - begin)

            begin = end

        ind = dataset.indices.to(device)
        err_by_len = err.cumsum(dim=0)[ind - 1]
        err_by_len[ind == 0] = 0
        err_by_len = err_by_len.diff()

        seq_err_by_len = seq_err.cumsum(dim=0)[ind - 1]
        seq_err_by_len[ind == 0] = 0
        seq_err_by_len = seq_err_by_len.diff().float()

        nb_by_len = ind.diff()
        nb_by_len[nb_by_len == 0] = 1
        err_by_len /= nb_by_len
        seq_err_by_len /= nb_by_len
        if special:
            spe_err /= end

            return err_by_len, seq_err_by_len, spe_err
        return err_by_len, seq_err_by_len

    @staticmethod
    def eval_spe_tok_err(pred):
        """
        Compute special token syntaxic error.

        Parameters
        ----------
        pred: torch.Tensor
            predictions of CoT with correct prefix.

        Returns
        -------
        bos_err: float
            number of `begin of sentence` syntaxic error.
        eoi_err: float
            number of `end of input` syntaxic error.
        eos_err: float
            number of `end of sentence` syntaxic error.
        """

        eos_ind = (pred == 2).int()
        first_eos = eos_ind.argmax(dim=-1)
        nb_eos = eos_ind.sum(dim=-1)

        eos_err = (first_eos + nb_eos) != 18
        bos_err = (pred == 0).int().sum(dim=-1) != 0
        eoi_err = (pred == 1).int().sum(dim=-1) != 1

        eos_err = eos_err.float().mean()
        bos_err = bos_err.float().mean()
        eoi_err = eoi_err.float().mean()

        return bos_err, eoi_err, eos_err


class AttentionEval:
    def __init__(self, lengths):
        meaning = []
        for leng in lengths:
            meaning += [
                f"attn0_inv_{leng}",
                f"attn1_inv_{leng}",
                f"attn0_peaky_{leng}",
                f"attn0_peaky_{leng}",
                f"attn1_peaky_{leng}",
                f"attn1_peaky_{leng}",
            ]
        self.meaning = [f"train_{stri}" for stri in meaning] + [f"train_{stri}" for stri in meaning]
        self.eval_dim = len(self.meaning)

    def __call__(self, model, trainset, testset):
        res = []
        for dataset in [trainset, testset]:
            sequences = dataset.data
            _, attns = model(sequences, verbose=True)
            attn_inv, attn_peaky = attention_metrics(sequences, attns)
            res.append(torch.hstack((attn_inv, attn_peaky)).flatten())

        return torch.hstack(res)
