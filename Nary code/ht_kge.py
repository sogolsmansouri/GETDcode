import torch
import torch.nn as nn
import torch.nn.functional as F


class GETD_HT(nn.Module):
    """
    Hierarchical-Tucker decoder (arity 3 or 4) – drop-in replacement for
    the Tensor-Ring-based GETD class.

    Parameters
    ----------
    d      : KG data object (must expose .entities, .relations, .train_data)
    d_e    : entity embedding dim
    d_r    : relation embedding dim
    k      : number of positions per fact (k-1 = arity)  → 4 for ternary,
             5 for quaternary
    ni     : latent mode dim at each internal HT node
    rank   : hierarchical rank r
    device : 'cuda' | 'cpu'
    """
    def __init__(self, d, d_e, d_r, k, ni, rank, device, **kw):
        super().__init__()
        self.device = torch.device(device)
        self.ary    = k - 1             # entity slots (3 or 4)

        # ------- embeddings -------------------------------------------------
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        nn.init.normal_(self.E.weight, 0., 1e-3)
        nn.init.normal_(self.R.weight, 0., 1e-3)

        # ------- HT cores ----------------------------------------------------
        r = rank
        self.U_root = nn.Parameter(torch.randn(len(d.relations), r, r,
                                               device=device) * 1e-3)
        self.U12 = nn.Parameter(torch.randn(r, ni, ni, device=device) * 1e-3)
        self.U3R = nn.Parameter(torch.randn(r, ni, ni, device=device) * 1e-3)
        self.U34 = (nn.Parameter(torch.randn(r, ni, ni, device=device) * 1e-3)
                    if self.ary == 4 else None)
        self.P    = nn.Parameter(torch.randn(r, d_e, device=device) * 1e-3)

        # ------- regular training extras ------------------------------------
        self.bne  = nn.BatchNorm1d(d_e)
        self.bnr  = nn.BatchNorm1d(d_r)
        self.bnw  = nn.BatchNorm1d(d_e)
        self.in_drop  = nn.Dropout(kw["input_dropout"])
        self.hid_drop = nn.Dropout(kw["hidden_dropout"])

    # --------------------------------------------------------------------- #
    #  Build W for the batch of relations (lazy, so no huge GPU tensor)     #
    # --------------------------------------------------------------------- #
    def _build_W(self, rel_ids: torch.Tensor):
        B   = rel_ids.size(0)
        d_r = self.R.embedding_dim
        d_e = self.E.embedding_dim

        Uroot = self.U_root[rel_ids]                # (B,r,r)

        if self.ary == 3:         # k = 4
            W = torch.einsum(
                'brs,ria,rjb,rkc->bijk',
                Uroot,
                self.U12,
                self.U3R,
                self.P.unsqueeze(1)                 # (r,1,d_e)
            ).view(B, d_r, d_e, d_e, d_e)
        else:                      # self.ary == 4   (k = 5)
            W = torch.einsum(
                'brs,ria,rjb,rkc,rlm->bijklm',
                Uroot,
                self.U12,
                self.U3R,
                self.U34,
                self.P.unsqueeze(1)
            ).view(B, d_r, d_e, d_e, d_e, d_e)

        return W

    # --------------------------------------------------------------------- #
    #  Forward                                                              #
    # --------------------------------------------------------------------- #
    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        """
        r_idx             : (B,)   tensor of relation IDs
        e_idx             : tuple of entity-ID tensors (length ary-1)
        miss_ent_domain   : which position is the target (1-based)
        returns           : pred (B, |E|),  None
        """
        B   = r_idx.size(0)
        d_r = self.R.embedding_dim
        d_e = self.E.embedding_dim

        # Relation gating
        r_emb  = self.bnr(self.R(r_idx))            # (B,d_r)
        W_full = self._build_W(r_idx) if W is None else W
        W_mat  = torch.bmm(r_emb.view(B,1,d_r),
                           W_full.view(B, d_r, -1)).view(B, *W_full.shape[2:])

        # Entity contraction
        if self.ary == 3:
            e2, e3 = (self.in_drop(self.bne(self.E(e))) for e in e_idx)
            if   miss_ent_domain == 1:
                vec = torch.einsum('bijk,bk,bj->bi', W_mat, e3, e2)
            elif miss_ent_domain == 2:
                vec = torch.einsum('bijk,bk,bi->bj', W_mat, e3, e2)
            else:
                vec = torch.einsum('bijk,bi,bj->bk', W_mat, e2, e3)

        else:  # arity==4
            e2, e3, e4 = (self.in_drop(self.bne(self.E(e))) for e in e_idx)
            if   miss_ent_domain == 1:
                vec = torch.einsum('bijkl,bk,bj,bl->bi', W_mat, e3, e2, e4)
            elif miss_ent_domain == 2:
                vec = torch.einsum('bijkl,bk,bi,bl->bj', W_mat, e3, e2, e4)
            elif miss_ent_domain == 3:
                vec = torch.einsum('bijkl,bi,bj,bl->bk', W_mat, e2, e3, e4)
            else:
                vec = torch.einsum('bijkl,bi,bj,bk->bl', W_mat, e2, e3, e4)

        vec  = self.hid_drop(self.bnw(vec))
        logits = vec @ self.E.weight.T
        return F.softmax(logits, dim=1), None
