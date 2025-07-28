import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tucker, tensor_train, tensor_ring
from tensorly.decomposition import tucker
from itertools import combinations
import string
import opt_einsum as oe  # Make sure this is imported

tl.set_backend('pytorch')
from opt_einsum import contract
#from tensorly_torch.decomposition import hierarchical_tucker


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return
    def forward(self, pred1, tar1):
        loss = F.binary_cross_entropy(pred1, tar1)
        return loss


import time
import logging
from itertools import combinations

import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class GETD_FC_opt_pos(nn.Module):
    """
    Fully-connected GETD model with on-the-fly contraction,
    optimized bond collapse, combined core+embed, profiling,
    and position-aware entity embeddings.
    Caches einsum equation strings to avoid runtime re-parsing.
    Profiling messages are written to 'profiling.log'.
    """
    def __init__(self, data, d_e, d_r, ni_list, rank_list, device, **kwargs):
        super().__init__()
        self.k = len(ni_list)
        assert self.k >= 2, "Need at least relation + entities"
        # Precompute edges and bond ranks
        self.edges = list(combinations(range(self.k), 2))
        self.bond_ranks = {edge: rank_list[i] for i, edge in enumerate(self.edges)}
        self.ni_list = ni_list

        # Embeddings
        self.E = nn.Embedding(len(data.entities), d_e, padding_idx=0).to(device)
        self.R = nn.Embedding(len(data.relations), d_r, padding_idx=0).to(device)
        # Random init
        # self.E.weight.data = 1e-3 * torch.randn_like(self.E.weight)
        # self.R.weight.data = 1e-3 * torch.randn_like(self.R.weight)
        self.E.weight.data = (1e-3 * torch.randn((len(data.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(data.relations), d_r), dtype=torch.float).to(device))
        # Position embeddings for each slot
        self.position = nn.Parameter(torch.randn(self.k, d_e) * 0.1)

        # Core tensors
        self.cores = nn.ParameterList()
        for i in range(self.k):
            shape = []
            for j in range(self.k):
                if i == j:
                    continue
                edge = (i, j) if i < j else (j, i)
                shape.append(self.bond_ranks[edge])
            shape.append(self.ni_list[i])
            G = nn.Parameter(
                torch.tensor(
                    np.random.uniform(-1e-1, 1e-1, tuple(shape)),
                    dtype=torch.float,
                    device=device
                )
            )
            self.cores.append(G)

        # Norms & dropout
        self.bnr = nn.BatchNorm1d(d_r)
        self.bne = nn.BatchNorm1d(d_e)
        self.bnw = nn.BatchNorm1d(d_e)
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.0))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.0))

        # Precompute einsum equations
        phys_letters = [chr(ord('A') + i) for i in range(self.k)]
        bond_letters = {e: chr(ord('a') + idx) for idx, e in enumerate(self.edges)}

        # Step1: fuse relation into core0
        p0 = phys_letters[0]
        b0 = [bond_letters[e] for e in self.edges if 0 in e]
        self.eq_fuse_rel = f"z{p0},{''.join(b0)+p0}->z{''.join(b0)}"

        # Step2: absorb known entities into T
        sub = 'z' + ''.join(b0)
        self.eq_absorb = {}
        for m in range(1, self.k):
            bm = [bond_letters[e] for e in self.edges if m in e]
            pm = phys_letters[m]
            eq = f"{sub},{''.join(bm)+pm},z{pm}->{sub}"
            self.eq_absorb[m] = eq

        # Step3: fuse missing core
        self.eq_missing = {}
        for m in range(1, self.k):
            bm = [bond_letters[e] for e in self.edges if m in e]
            pm = phys_letters[m]
            self.eq_missing[m] = f"{sub},{''.join(bm)+pm}->{sub+pm}"

    def forward(self, r_idx, e_idx_list, miss, W=None):
        torch.cuda.reset_peak_memory_stats()
        B = r_idx.size(0)

        # Step1: fuse relation
        start = time.time()
        R0 = self.input_dropout(self.bnr(self.R(r_idx)))  # (B, d_r)
        T = torch.einsum(self.eq_fuse_rel, R0, self.cores[0])
        mem1 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step1 fuse relation: time={time.time()-start:.4f}s, mem={mem1:.1f}MiB")

        # Step2: absorb known entities
        start = time.time()
        idx_known = 0
        for m in range(1, self.k):
            if m == miss:
                continue
            Em = self.E(e_idx_list[idx_known])  # (B, d_e)
            # add position embedding for slot m
            Em = Em + self.position[m]
            Em = self.input_dropout(self.bne(Em))
            eq = self.eq_absorb[m]
            T = torch.einsum(eq, T, self.cores[m], Em)
            idx_known += 1
        mem2 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step2 fuse+absorb entities: time={time.time()-start:.4f}s, mem={mem2:.1f}MiB")

        # Step3: fuse missing core
        start = time.time()
        eqm = self.eq_missing[miss]
        T = torch.einsum(eqm, T, self.cores[miss])
        mem3 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step3 fuse missing core: time={time.time()-start:.4f}s, mem={mem3:.1f}MiB")

        # Step4: collapse bonds
        start = time.time()
        S = T.sum(dim=tuple(range(1, T.dim()-1)))  # (B, n_i[miss])
        mem4 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step4 collapse bonds: time={time.time()-start:.4f}s, mem={mem4:.1f}MiB")

        # Step5: final BN, dropout, projection and score
        start = time.time()
        # add position embedding for missing slot before BN
        S = S + self.position[miss]
        out = self.hidden_dropout(self.bnw(S))  # (B, d_e)
        logits = out @ self.E.weight.t()         # (B, num_entities)
        mem5 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step5 final layers: time={time.time()-start:.4f}s, mem={mem5:.1f}MiB")

        return logits, W

import torch
import torch.nn.functional as F
import numpy as np
from itertools import combinations

import math
import torch
import torch.nn.functional as F

import math
import torch
import torch.nn.functional as F
from itertools import combinations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GETD_FC_chunked(nn.Module):
    def __init__(
        self, data, d_e, d_r, ni_list, rank_list,
        device, chunks=30, input_dropout=0.1, hidden_dropout=0.1
    ):
        super().__init__()
        self.device = device
        self.chunks = chunks
        self.ni_list = ni_list
        self.rank_list = rank_list

        # Embeddings (created on CPU to avoid premature GPU OOM)
        self.E = nn.Embedding(len(data.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(data.relations), d_r, padding_idx=0)
        # Batch norms + dropouts
        self.bnr = nn.BatchNorm1d(d_r)
        self.bne = nn.BatchNorm1d(d_e)
        self.bnw = nn.BatchNorm1d(max(ni_list[1:]))
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.hidden_dropout = nn.Dropout(p=hidden_dropout)

        # Three-way cores (bond-ranks, moved to GPU later)
        n0,n1,n2,n3 = ni_list
        r0,r1,r2,r3,r4,r5 = rank_list
        self.cores = nn.ParameterList([
            nn.Parameter(torch.randn(r0, r1, r2, n0)),
            nn.Parameter(torch.randn(r0, r3, r4, n1)),
            nn.Parameter(torch.randn(r1, r3, r5, n2)),
            nn.Parameter(torch.randn(r2, r4, r5, n3)),
        ])

        # Clear leftover cache then move modules & parameters to GPU
        torch.cuda.empty_cache()
        self.to(device)

    def forward(self, r_idx, e_idx, miss, W=None):
        device = r_idx.device
        B = r_idx.size(0)

        # Unpack dims & cores
        G0, G1, G2, G3 = self.cores
        n0,n1,n2,n3 = self.ni_list
        i_size, m_size = self.rank_list[0], self.rank_list[5]

        # Boundary helper
        def boundaries(n):
            step = math.ceil(n / self.chunks)
            cuts = list(range(0, n, step))
            return cuts + [n]

        # Precompute all slice bounds
        a_bounds = boundaries(n0)
        b_bounds = boundaries(n1)
        c_bounds = boundaries(n2)
        d_bounds = boundaries(n3)
        i_bounds = boundaries(i_size)
        m_bounds = boundaries(m_size)

        # Embeddings + batchnorm + dropout
        r_emb = self.input_dropout(self.bnr(self.R(r_idx)))  # (B,dr)
        e2    = self.input_dropout(self.bne(self.E(e_idx[0])))# (B,de)
        e3    = self.input_dropout(self.bne(self.E(e_idx[1])))# (B,de)

        # Prepare logits
        out_size = [n1, n2, n3][miss-1]
        logits   = torch.zeros(B, out_size, device=device)

        # Nested loops: modes + rank-chunks
        for ia in range(len(a_bounds)-1):
            a0,a1 = a_bounds[ia], a_bounds[ia+1]
            G0_a  = G0[..., a0:a1]
            r_sub = r_emb[:, a0:a1]

            for ib in range(len(b_bounds)-1):
                b0,b1 = b_bounds[ib], b_bounds[ib+1]
                G1_b  = G1[..., b0:b1]

                # Chunk & sum over bond-rank i
                K01_sum = None
                for ir in range(len(i_bounds)-1):
                    i0,i1 = i_bounds[ir], i_bounds[ir+1]
                    part01 = torch.einsum(
                        'ijka,inlb->jkanlb',
                        G0_a[i0:i1,...], G1_b[i0:i1,...]
                    )
                    K01_sum = part01 if K01_sum is None else K01_sum + part01
                    del part01; torch.cuda.empty_cache()

                for ic in range(len(c_bounds)-1):
                    c0,c1 = c_bounds[ic], c_bounds[ic+1]
                    G2_c  = G2[..., c0:c1]

                    for idd in range(len(d_bounds)-1):
                        d0,d1 = d_bounds[idd], d_bounds[idd+1]
                        G3_d  = G3[..., d0:d1]

                        # Chunk & sum over bond-rank m
                        K23_sum = None
                        for im in range(len(m_bounds)-1):
                            m0,m1 = m_bounds[im], m_bounds[im+1]
                            part23 = torch.einsum(
                                'jnmc,klmd->jnckld',
                                G2_c[..., m0:m1,:], G3_d[..., m0:m1,:]
                            )
                            K23_sum = part23 if K23_sum is None else K23_sum + part23
                            del part23; torch.cuda.empty_cache()

                        # Fuse partial sums -> 4D block
                        block   = torch.einsum('jkanlb,jnckld->abcd', K01_sum, K23_sum)
                        del K23_sum; torch.cuda.empty_cache()

                        # Fuse relation
                        partial = torch.einsum('Ba,abcd->Bbcd', r_sub, block)
                        del block; torch.cuda.empty_cache()

                        # Contract known entities -> logits
                        if miss == 1:
                            ec, ed = e2[:, c0:c1], e3[:, d0:d1]
                            logits[:, b0:b1] += torch.einsum(
                                'Bbcd,Bc,Bd->Bb', partial, ec, ed
                            )
                        elif miss == 2:
                            eb, ed = e2[:, b0:b1], e3[:, d0:d1]
                            logits[:, c0:c1] += torch.einsum(
                                'Bbcd,Bb,Bd->Bc', partial, eb, ed
                            )
                        else:
                            eb, ec = e2[:, b0:b1], e3[:, c0:c1]
                            logits[:, d0:d1] += torch.einsum(
                                'Bbcd,Bb,Bc->Bd', partial, eb, ec
                            )
                        del partial; torch.cuda.empty_cache()

                # Clean up K01_sum after using in all c,d
                del K01_sum; torch.cuda.empty_cache()

        # Final batchnorm + dropout + final projection
        logits = self.hidden_dropout(self.bnw(logits))
        out    = torch.mm(logits, self.E.weight.t())
        pred   = F.softmax(out, dim=1)

        # Cleanup large tensors before returning
        del G0, G1, G2, G3, r_emb, e2, e3, logits; torch.cuda.empty_cache()
        return pred, W

class GETD_new_FC(torch.nn.Module):
    def __init__(self, d, d_e, d_r, ni_list, rank_list, device, chunks=3, **kwargs):
        super().__init__()
        self.k = len(ni_list)
        assert self.k >= 2, "Need at least relation + entities"
        assert len(rank_list) == self.k*(self.k-1)//2, \
            f"need {self.k*(self.k-1)//2} bond ranks, got {len(rank_list)}"

        # Precompute edges and bond ranks
        self.edges = list(combinations(range(self.k), 2))
        self.bond_ranks = {edge: rank_list[i] for i, edge in enumerate(self.edges)}
        self.ni_list = ni_list
        self.chunks  = chunks
        self.rank_list = rank_list
        # Embeddings
        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        
        self.cores = torch.nn.ParameterList()
        for i in range(self.k):
            shape = []
            for j in range(self.k):
                if i == j:
                    continue
                edge = (i, j) if i < j else (j, i)
                shape.append(self.bond_ranks[edge])
            shape.append(self.ni_list[i])
            # initialize with numpy uniform of shape tuple
            size_tuple = tuple(shape)
            G = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-1e-1, 1e-1, size_tuple),
                    dtype=torch.float,
                    requires_grad=True
                ).to(device)
            )
            self.cores.append(G)
        
        self.loss = MyLoss()
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm1d(d_e)
        self.bnr = torch.nn.BatchNorm1d(d_r)
        self.bnw = torch.nn.BatchNorm1d(d_e)
        self.ary = len(d.train_data[0]) - 1
 

    def forward(self, r_idx, e_idx, miss, W=None):
        """
        r_idx: (B,)                  # relation indices per batch
        e_idx: list of two (B,)     # entity indices for the two known entities
        miss:   int in {1,2,3}      # which entity to predict
        """
        device = r_idx.device
        B      = r_idx.size(0)
        de     = self.E.embedding_dim
        dr     = self.R.embedding_dim

        # unpack our 4 three-way cores
        G0, G1, G2, G3 = self.cores    # shapes: G0[i,j,k,a], G1[i,n,l,b], G2[j,n,m,c], G3[k,l,m,d]
        n0,n1,n2,n3   = self.ni_list   # num entities per slot
        # bond ranks: i = self.rank_list[0], m = self.rank_list[5]
        i_size = self.rank_list[0]
        m_size = self.rank_list[5]

        # helper to split any dimension into ~self.chunks pieces
        def boundaries(n):
            step = math.ceil(n / self.chunks)
            cuts = list(range(0, n, step))
            return cuts + [n]

        # compute slice boundaries for each mode and both rank dims
        a_bounds = boundaries(n0)
        b_bounds = boundaries(n1)
        c_bounds = boundaries(n2)
        d_bounds = boundaries(n3)
        i_bounds = boundaries(i_size)
        m_bounds = boundaries(m_size)

        # compute embeddings + dropouts
        r_emb = self.input_dropout(self.bnr(self.R(r_idx)))  # (B, dr)
        e2    = self.input_dropout(self.bne(self.E(e_idx[0])))# (B, de)
        e3    = self.input_dropout(self.bne(self.E(e_idx[1])))# (B, de)

        # final logits buffer
        out_size = [n1, n2, n3][miss-1]
        logits   = torch.zeros(B, out_size, device=device)

        # four nested loops over mode slices, plus two loops over rank slices
        for ia in range(len(a_bounds)-1):
            a0, a1 = a_bounds[ia], a_bounds[ia+1]
            # slice core-mode a
            G0_a = G0[..., a0:a1]             # [i, j, k, a_slice]
            r_sub = r_emb[:, a0:a1]           # (B, a_slice)

            for ib in range(len(b_bounds)-1):
                b0, b1 = b_bounds[ib], b_bounds[ib+1]
                G1_b = G1[..., b0:b1]         # [i, n, l, b_slice]

                # chunk over bond-rank i for K01
                K01_sum = None
                for ir in range(len(i_bounds)-1):
                    i0, i1 = i_bounds[ir], i_bounds[ir+1]
                    G0_ir = G0_a[i0:i1, ...]   # [i_slice, j, k, a_slice]
                    G1_ir = G1_b[i0:i1, ...]   # [i_slice, n, l, b_slice]
                    part01 = torch.einsum('ijka,inlb->jkanlb', G0_ir, G1_ir)
                    K01_sum = part01 if K01_sum is None else K01_sum + part01

                for ic in range(len(c_bounds)-1):
                    c0, c1 = c_bounds[ic], c_bounds[ic+1]
                    G2_c = G2[..., c0:c1]     # [j, n, m, c_slice]

                    for idd in range(len(d_bounds)-1):
                        d0, d1 = d_bounds[idd], d_bounds[idd+1]
                        G3_d = G3[..., d0:d1]   # [k, l, m, d_slice]

                        # chunk over bond-rank m for K23
                        K23_sum = None
                        for im in range(len(m_bounds)-1):
                            m0, m1 = m_bounds[im], m_bounds[im+1]
                            G2_cm = G2_c[..., m0:m1, :]  # [j, n, m_slice, c_slice]
                            G3_dm = G3_d[..., m0:m1, :]  # [k, l, m_slice, d_slice]
                            part23 = torch.einsum('jnmc,klmd->jnckld', G2_cm, G3_dm)
                            K23_sum = part23 if K23_sum is None else K23_sum + part23

                        # fuse K01 + K23 -> [a_slice, b_slice, c_slice, d_slice]
                        block = torch.einsum('jkanlb,jnckld->abcd', K01_sum, K23_sum)

                        # fuse relation embedding -> [B, b_slice, c_slice, d_slice]
                        partial = torch.einsum('Ba,abcd->Bbcd', r_sub, block)

                        # contract known entity embeddings into logits
                        if   miss == 1:
                            ec = e2[:, c0:c1]; ed = e3[:, d0:d1]
                            logits[:, b0:b1] += torch.einsum('Bbcd,Bc,Bd->Bb', partial, ec, ed)
                        elif miss == 2:
                            eb = e2[:, b0:b1]; ed = e3[:, d0:d1]
                            logits[:, c0:c1] += torch.einsum('Bbcd,Bb,Bd->Bc', partial, eb, ed)
                        else:
                            eb = e2[:, b0:b1]; ec = e3[:, c0:c1]
                            logits[:, d0:d1] += torch.einsum('Bbcd,Bb,Bc->Bd', partial, eb, ec)

        # final BN + dropout + softmax over entity embedding matrix
        logits = self.hidden_dropout(self.bnw(logits))
        x      = torch.mm(logits, self.E.weight.t())  # (B, #entities)

        pred   = F.softmax(x, dim=1)
        return pred, W


    # def forward(self, r_idx, e_idx, miss, W=None):
    #     """
    #     r_idx: (B,)
    #     e_idx: list of two LongTensors, each (B,)
    #     miss:   int 1/2/3 = which entity to predict
    #     """
    #     device = r_idx.device
    #     B      = r_idx.size(0)
    #     de     = self.E.embedding_dim
    #     dr     = self.R.embedding_dim
    #     G0,G1,G2,G3 = self.cores     # shapes: see above
    #     n0,n1,n2,n3 = self.ni_list         # for clarity; you could read from self.ni_list

    #     # Embeddings
    #     r_emb = self.bnr(self.R(r_idx))                            # (B, dr)
    #     e2    = self.bne(self.E(e_idx[0]))                          # (B, de)
    #     e3    = self.bne(self.E(e_idx[1]))                          # (B, de)
    #     r_emb = self.input_dropout(r_emb)
    #     e2    = self.input_dropout(e2)
    #     e3    = self.input_dropout(e3)

    #     # Allocate the final logits buffer: shape (B, max domain size)
    #     out_size = [n1,n2,n3][miss-1]
    #     logits   = torch.zeros(B, out_size, device=device)

    #     # Precompute slice boundaries for each mode
    #     def boundaries(n):
    #         step = math.ceil(n / self.chunks)
    #         cuts = list(range(0, n, step))
    #         return cuts + [n]  # ensure we end at n

    #     a_bounds = boundaries(n0)  # e.g. [0,6,12,…,90]
    #     b_bounds = boundaries(n1)
    #     c_bounds = boundaries(n2)
    #     d_bounds = boundaries(n3)

    #     # Now the four‐deep loops
    #     for ia in range(len(a_bounds)-1):
    #         a0, a1 = a_bounds[ia],   a_bounds[ia+1]
    #         # slice G0,G1 on their a‐index
    #         G0_sub = G0[..., a0:a1]             # shape [90,90,90, a_slice]
    #         r_sub  = r_emb[:, a0:a1]            # (B, a_slice)

    #         for ib in range(len(b_bounds)-1):
    #             b0, b1 = b_bounds[ib],   b_bounds[ib+1]
    #             G1_sub = G1[..., b0:b1]         # shape [90,90,90, b_slice]

    #             # fuse G0_sub + G1_sub over the i‐index
    #             # result: [j,k,a_slice,b_slice]
    #             K01 = torch.einsum('ijka,inlb->jkanlb',
    #                                G0_sub, G1_sub)

    #             for ic in range(len(c_bounds)-1):
    #                 c0, c1 = c_bounds[ic],   c_bounds[ic+1]
    #                 G2_sub = G2[..., c0:c1]     # [90,90,90, c_slice]

    #                 for idd in range(len(d_bounds)-1):
    #                     d0, d1 = d_bounds[idd], d_bounds[idd+1]
    #                     G3_sub = G3[..., d0:d1]   # [90,90,90, d_slice]

    #                     # fuse G2_sub + G3_sub over m‐index
    #                     # → [j,k,c_slice,d_slice]
    #                     K23 = torch.einsum('jnmc,klmd->jnckld',
    #                                        G2_sub, G3_sub)

    #                     # fuse K01 + K23 over (j,k)
    #                     # → block [a_slice, b_slice, c_slice, d_slice]
    #                     block = torch.einsum('jkanlb,jnckld->abcd',
    #                                          K01, K23)

    #                     # now fuse in the relation embedding r_sub (over a)
    #                     # → partial [B, b_slice, c_slice, d_slice]
    #                     partial = torch.einsum('Ba,abcd->Bbcd',
    #                                            r_sub, block)

    #                     # finally, depending on which domain is missing,
    #                     # contract the known embeddings to get [B, slice_dim]
    #                     if miss == 1:
    #                         # predicting b: known c→e2, d→e3
    #                         e_c = e2[:, c0:c1]
    #                         e_d = e3[:, d0:d1]
    #                         # contract to get [B, b_slice]
    #                         part_logit = torch.einsum('Bbcd, Bc, Bd -> Bb',
    #                                                   partial, e_c, e_d)
    #                         logits[:, b0:b1] += part_logit

    #                     elif miss == 2:
    #                         # predicting c: known b→e2, d→e3
    #                         e_b = e2[:, b0:b1]
    #                         e_d = e3[:, d0:d1]
    #                         part_logit = torch.einsum('Bbcd, Bb, Bd -> Bc',
    #                                                   partial, e_b, e_d)
    #                         logits[:, c0:c1] += part_logit

    #                     else:  # miss==3
    #                         # predicting d: known b→e2, c→e3
    #                         e_b = e2[:, b0:b1]
    #                         e_c = e3[:, c0:c1]
    #                         part_logit = torch.einsum('Bbcd, Bb, Bc -> Bd',
    #                                                   partial, e_b, e_c)
    #                         logits[:, d0:d1] += part_logit

    #     # final BN + dropout
    #     logits = self.bnw(logits)
    #     logits = self.hidden_dropout(logits)

    #     # and produce raw scores for all entities:
    #     # x = logits @ self.E.weight.t()   # (B, #entities)
    #     # return x
    #     x = torch.mm(logits, self.E.weight.transpose(1, 0))

    #     pred = F.softmax(x, dim=1)

    #     return pred, W


    # def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
    #     de = self.E.weight.shape[1]
    #     dr = self.R.weight.shape[1]
    #     G0, G1, G2, G3 = self.cores
        
        
 
    #     e2, e3 = self.bne(self.E(e_idx[0])), self.bne(self.E(e_idx[1]))
    #     e2, e3 = self.input_dropout(e2), self.input_dropout(e3)
        
    #     if self.k == 4:
            
           
    #         K01 = torch.einsum('ijka,inlb->jkanlb', G0, G1)
    #         K23 = torch.einsum('jnmc,klmd->jnckld', G2, G3)
    #         W0 = torch.einsum('jkanlb,jnckld->abcd', K01, K23)
    #         W = W0.view(dr, de, de, de)
    #         r = self.bnr(self.R(r_idx))
    #         W_mat = torch.mm(r, W.view(r.size(1), -1))
    #         W_mat = W_mat.view(-1, de, de, de)
            
    #         if miss_ent_domain == 1:
    #             W_mat1 = torch.einsum('ijkl,il,ik->ij', W_mat, e3, e2)
    #         elif miss_ent_domain == 2:
    #             W_mat1 = torch.einsum('ijkl,il,ij->ik', W_mat, e3, e2)
    #         elif miss_ent_domain == 3:
    #             W_mat1 = torch.einsum('ijkl,ij,ik->il', W_mat, e2, e3)
    #         torch.cuda.empty_cache()  
    #         W_mat1 = self.bnw(W_mat1)
    #         W_mat1 = self.hidden_dropout(W_mat1)
    #         x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))

    #         pred = F.softmax(x, dim=1)

    #         return pred, W





    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GETD_FC_opt(nn.Module):
    """
    Fully-connected GETD model with on-the-fly contraction,
    optimized bond collapse, combined core+embed, and profiling.
    Caches einsum equation strings to avoid runtime re-parsing.
    Profiling messages are written to 'profiling.log'.
    """
    def __init__(self, data, d_e, d_r, ni_list, rank_list, device, **kwargs):
        super().__init__()
        self.k = len(ni_list)
        assert self.k >= 2, "Need at least relation + entities"
        # Precompute edges and bond ranks
        self.edges = list(combinations(range(self.k), 2))
        self.bond_ranks = {edge: rank_list[i] for i, edge in enumerate(self.edges)}
        self.ni_list = ni_list

        # Embeddings
        self.E = nn.Embedding(len(data.entities), d_e, padding_idx=0).to(device)
        self.R = nn.Embedding(len(data.relations), d_r, padding_idx=0).to(device)
        # Random init
        # self.E.weight.data = 1e-3 * torch.randn_like(self.E.weight)
        # self.R.weight.data = 1e-3 * torch.randn_like(self.R.weight)
        self.E.weight.data = (1e-3 * torch.randn((len(data.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(data.relations), d_r), dtype=torch.float).to(device))
        # Core tensors
        self.cores = nn.ParameterList()
        for i in range(self.k):
            shape = []
            for j in range(self.k):
                if i == j:
                    continue
                edge = (i, j) if i < j else (j, i)
                shape.append(self.bond_ranks[edge])
            shape.append(self.ni_list[i])
            G = nn.Parameter(
                torch.tensor(
                    np.random.uniform(-1e-1, 1e-1, tuple(shape)),
                    dtype=torch.float,
                    device=device
                )
            )
            self.cores.append(G)

        # Norms & dropout
        self.bnr = nn.BatchNorm1d(d_r)
        self.bne = nn.BatchNorm1d(d_e)
        self.bnw = nn.BatchNorm1d(d_e)
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.0))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.0))

        # LOSS placeholder
        # self.loss = MyLoss()

        # Precompute einsum equations
        phys_letters = [chr(ord('A') + i) for i in range(self.k)]
        bond_letters = {e: chr(ord('a') + idx) for idx, e in enumerate(self.edges)}

        # Step1: fuse relation into core0
        p0 = phys_letters[0]
        b0 = [bond_letters[e] for e in self.edges if 0 in e]
        self.eq_fuse_rel = f"z{p0},{''.join(b0)+p0}->z{''.join(b0)}"

        # Step2: absorb known entities into T
        sub = 'z' + ''.join(b0)
        self.eq_absorb = {}
        for m in range(1, self.k):
            bm = [bond_letters[e] for e in self.edges if m in e]
            pm = phys_letters[m]
            eq = f"{sub},{''.join(bm)+pm},z{pm}->{sub}"
            self.eq_absorb[m] = eq

        # Step3: fuse missing core
        self.eq_missing = {}
        for m in range(1, self.k):
            bm = [bond_letters[e] for e in self.edges if m in e]
            pm = phys_letters[m]
            self.eq_missing[m] = f"{sub},{''.join(bm)+pm}->{sub+pm}"

    def forward(self, r_idx, e_idx_list, miss, W=None):
        torch.cuda.reset_peak_memory_stats()
        B = r_idx.size(0)

        # Step1: fuse relation
        start = time.time()
        R0 = self.input_dropout(self.bnr(self.R(r_idx)))  # (B, d_r)
        T = torch.einsum(self.eq_fuse_rel, R0, self.cores[0])
        mem1 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step1 fuse relation: time={time.time()-start:.4f}s, mem={mem1:.1f}MiB")
        sub = None  # not needed further

        # Step2: absorb known entities
        start = time.time()
        idx_known = 0
        for m in range(1, self.k):
            if m == miss:
                continue
            Em = self.input_dropout(self.bne(self.E(e_idx_list[idx_known])))  # (B, d_e)
            eq = self.eq_absorb[m]
            T = torch.einsum(eq, T, self.cores[m], Em)
            idx_known += 1
        mem2 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step2 fuse+absorb entities: time={time.time()-start:.4f}s, mem={mem2:.1f}MiB")

        # Step3: fuse missing core
        start = time.time()
        eqm = self.eq_missing[miss]
        T = torch.einsum(eqm, T, self.cores[miss])
        mem3 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step3 fuse missing core: time={time.time()-start:.4f}s, mem={mem3:.1f}MiB")

        # Step4: collapse bonds
        start = time.time()
        S = T.sum(dim=tuple(range(1, T.dim()-1)))  # (B, n_i[miss])
        mem4 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step4 collapse bonds: time={time.time()-start:.4f}s, mem={mem4:.1f}MiB")

        # Step5: final BN, dropout, softmax
        start = time.time()
        out = self.hidden_dropout(self.bnw(S))  # (B, d_e)
        logits = out @ self.E.weight.t()         # (B, num_entities)
        mem5 = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(f"Step5 final layers: time={time.time()-start:.4f}s, mem={mem5:.1f}MiB")

        return logits, W
import torch
import torch.nn as nn
import torch.nn.functional as F

class GETD_FC_pos(nn.Module):
    """
    Pure-FC tensor decomposition, star topology, position/role/edge gating,
    with optional auxiliary slot and relation prediction losses.
    Supports k=4 (one relation, 3 entities) with one entity missing (i.e., 2 known entities per query).
    """
    def __init__(self, d, edim, rdim, ni_list, rank_list, device,
                 input_dropout=0.1, hidden_dropout=0.1, use_aux=False, aux_alpha=0.1, aux_beta=0.1):
        super().__init__()
        self.device = device
        self.arity = len(ni_list)
        self.num_entities = len(d.entities)
        self.num_relations = len(d.relations)
        self.edim = edim
        self.rdim = rdim
        self.rank = rank_list[0]
        self.use_aux = use_aux
        self.aux_alpha = aux_alpha
        self.aux_beta = aux_beta

        # Slot-specific entity embeddings
        self.E = nn.ModuleList([nn.Embedding(self.num_entities, edim) for _ in range(self.arity)])
        self.R = nn.Embedding(self.num_relations, rdim)
        self.position = nn.Parameter(torch.randn(self.arity, edim) * 0.1)
        self.edge_gates = nn.Parameter(torch.zeros(self.arity))
        self.rank_weights = nn.ParameterList([nn.Parameter(torch.ones(self.rank)) for _ in range(self.arity)])
        self.C_entity = nn.ParameterList([
            nn.Parameter(torch.randn(self.rank, edim) * 0.1) for _ in range(self.arity)
        ])
        # Only (rank, rank, rdim) for arity=4 with one entity missing!
        self.C_relation = nn.Parameter(torch.randn(self.rank, self.rank, self.rdim) * 0.1)

        self.E_pred = nn.Embedding(self.num_entities, edim)
        self.proj = nn.Linear(edim, rdim, bias=False)
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

        # Auxiliary heads for slot and relation prediction
        if self.use_aux:
            self.role_heads = nn.ModuleList([nn.Linear(edim, self.arity) for _ in range(self.arity)])
            self.rel_head = nn.Linear(rdim, self.num_relations)

    def forward(self, r_idx, e_idx_list, missing_slot, W=None, return_aux=False):
        """
        r_idx: (B,)
        e_idx_list: list of (B,) entity idx, for the 2 known slots (k=4, one missing, so len=2)
        missing_slot: 1-based index (1 = first entity)
        Returns logits (B, num_entities), None [, aux_loss if return_aux]
        """
        B = r_idx.size(0)
        arity = self.arity
        device = r_idx.device

        # Which entity slots are present? (model expects slots 1..arity-1, missing_slot is not present)
        entity_slots = [slot for slot in range(1, arity) if slot != missing_slot]

        # Build entity embeddings (+ position + dropout), in correct slot order
        e_vecs = []
        role_logits = []
        for i, slot in enumerate(entity_slots):
            e = self.E[slot](e_idx_list[i]) + self.position[slot]
            e = self.input_dropout(e)
            e_vecs.append(e)
            if self.use_aux:
                role_logits.append(self.role_heads[slot](e))

        # Relation embedding
        r = self.R(r_idx)
        r = self.input_dropout(r)
        if self.use_aux:
            rel_logits = self.rel_head(r)

        # Contract with entity cores, apply gate/weights
        H = []
        for idx, slot in enumerate(entity_slots):
            g = torch.sigmoid(self.edge_gates[slot])
            w = F.softmax(self.rank_weights[slot], dim=0)
            C = self.C_entity[slot] * w[:, None]
            C = g * C
            h = torch.matmul(e_vecs[idx], C.T)
            H.append(h)  # Each (B, rank)

        # Main link prediction (over all entities in missing slot)
        logits = []
        for ent in range(self.num_entities):
            # Candidate for missing slot
            e_cand = self.E[missing_slot](
                torch.full((B,), ent, dtype=torch.long, device=device)
            ) + self.position[missing_slot]
            e_cand = self.input_dropout(e_cand)
            # Contract: build h_list, placing candidate in the missing slot's position
            h_list = []
            h_slot_iter = iter(H)
            for slot in range(1, arity):
                if slot == missing_slot:
                    g = torch.sigmoid(self.edge_gates[slot])
                    w = F.softmax(self.rank_weights[slot], dim=0)
                    C = self.C_entity[slot] * w[:, None]
                    C = g * C
                    h = torch.matmul(e_cand, C.T)
                    h_list.append(h)
                else:
                    h_list.append(next(h_slot_iter))
            # For arity=4 with 2 known entities, always do: 'bi,bj,ijk->bk'
            result = torch.einsum('bi,bj,ijk->bk', h_list[0], h_list[1], self.C_relation)
            # Fuse with relation embedding
            score = (result * r).sum(dim=1)  # (B,)
            logits.append(score)
        logits = torch.stack(logits, dim=1)  # (B, num_entities)

        # Auxiliary loss if requested
        if self.use_aux and return_aux:
            # Dummy targets for now (replace with real if available)
            true_roles = torch.zeros((B, arity-1), dtype=torch.long, device=device)
            true_rel = torch.zeros(B, dtype=torch.long, device=device)
            role_loss = sum(F.cross_entropy(role_logits[i], true_roles[:, entity_slots[i]-1]) for i in range(len(entity_slots)))
            rel_loss = F.cross_entropy(rel_logits, true_rel)
            aux_loss = self.aux_alpha * role_loss + self.aux_beta * rel_loss
            return logits, None, aux_loss

        return logits, None

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from itertools import combinations
import torch.nn.init as init
# Configure logging to file
logging.basicConfig(
    filename='profiling.log',
    filemode='a',  # append mode
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

class GETD_FC(nn.Module):
    """
    Fully-connected GETD model with on-the-fly contraction,
    optimized bond collapse, combined core+embed, and profiling.
    Profiling messages are written to 'profiling.log'.
    """
    def __init__(self, data, d_e, d_r, ni_list, rank_list, device, **kwargs):
        super().__init__()
        self.k = len(ni_list)
        assert self.k >= 2, "Need at least relation + entities"
        assert len(rank_list) == self.k*(self.k-1)//2, \
            f"need {self.k*(self.k-1)//2} bond ranks, got {len(rank_list)}"

        # Precompute edges and bond ranks
        self.edges = list(combinations(range(self.k), 2))
        self.bond_ranks = {edge: rank_list[i] for i, edge in enumerate(self.edges)}
        self.ni_list = ni_list

        # Embeddings
        self.E = nn.Embedding(len(data.entities), d_e, padding_idx=0).to(device)
        self.R = nn.Embedding(len(data.relations), d_r, padding_idx=0).to(device)
        # nn.init.normal_(self.E.weight, std=1e-3)
        # nn.init.normal_(self.R.weight, std=1e-3)
        self.E.weight.data = (1e-3 * torch.randn((len(data.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(data.relations), d_r), dtype=torch.float).to(device))
        # Core tensors
        # self.cores = nn.ParameterList()
        # for i in range(self.k):
        #     dims = [self.bond_ranks[(i,j)] if i<j else self.bond_ranks[(j,i)]
        #             for j in range(self.k) if j!=i]
        #     dims.append(self.ni_list[i])
        #     G = nn.Parameter(torch.empty(*dims, device=device).uniform_(-1e-1,1e-1))
        #     self.cores.append(G)
        
        self.cores = torch.nn.ParameterList()
        for i in range(self.k):
            shape = []
            for j in range(self.k):
                if i == j:
                    continue
                edge = (i, j) if i < j else (j, i)
                shape.append(self.bond_ranks[edge])
            shape.append(self.ni_list[i])
            # initialize with numpy uniform of shape tuple
            size_tuple = tuple(shape)
            G = torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-1e-1, 1e-1, size_tuple),
                    dtype=torch.float,
                    requires_grad=True
                ).to(device)
            )
            self.cores.append(G)
            
            # for G in self.cores:
            #     if G.dim() > 1:
            #         init.xavier_uniform_(G)


        # Norms & dropout
        self.bnr = nn.BatchNorm1d(d_r)
        self.bne = nn.BatchNorm1d(d_e)
        self.bnw = nn.BatchNorm1d(d_e)
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout",0.0))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout",0.0))
        self.loss = MyLoss()

    def forward(self, r_idx, e_idx_list, miss, W=None):
        torch.cuda.reset_peak_memory_stats()
        B = r_idx.size(0)

        # Subscript letters
        bond_letters = {e: chr(ord('a')+i) for i,e in enumerate(self.edges)}
        phys_letters = [chr(ord('A')+m) for m in range(self.k)]

        # Step1: Fuse relation into core0
        start = time.time()
        R0 = self.input_dropout(self.bnr(self.R(r_idx)))  # (B, c0)
        p0 = phys_letters[0]
        b0 = [bond_letters[e] for e in self.edges if 0 in e]
        eq0 = f"z{p0},{''.join(b0)+p0}->z{''.join(b0)}"
        T = torch.einsum(eq0, R0, self.cores[0])
        mem1 = torch.cuda.max_memory_allocated()/1024**2
        logging.info(f"Step1 fuse relation: time={time.time()-start:.4f}s, mem={mem1:.1f}MiB")
        sub = 'z' + ''.join(b0)

        # Step2: Combined fuse+absorb for each known entity
        start = time.time()
        idx_known = 0
        for m in range(1,self.k):
            if m==miss: continue
            Gm = self.cores[m]
            pm = phys_letters[m]
            bm = [bond_letters[e] for e in self.edges if m in e]
            Em = self.input_dropout(self.bne(self.E(e_idx_list[idx_known])))
            idx_known += 1
            eq = f"{sub},{''.join(bm)+pm},z{pm}->{sub}"
            T = torch.einsum(eq, T, Gm, Em)
        mem2 = torch.cuda.max_memory_allocated()/1024**2
        logging.info(f"Step2 fuse+absorb entities: time={time.time()-start:.4f}s, mem={mem2:.1f}MiB")

        # Step3: Fuse missing core
        start = time.time()
        Gm = self.cores[miss]
        pm = phys_letters[miss]
        bm = [bond_letters[e] for e in self.edges if miss in e]
        eqm = f"{sub},{''.join(bm)+pm}->{sub+pm}"
        T = torch.einsum(eqm, T, Gm)
        mem3 = torch.cuda.max_memory_allocated()/1024**2
        logging.info(f"Step3 fuse missing core: time={time.time()-start:.4f}s, mem={mem3:.1f}MiB")
        sub += pm

        # Step4: Collapse all bond dims at once
        start = time.time()
        S = T.sum(dim=tuple(range(1, T.dim()-1)))  # (B, c_m)
        mem4 = torch.cuda.max_memory_allocated()/1024**2
        logging.info(f"Step4 collapse bonds: time={time.time()-start:.4f}s, mem={mem4:.1f}MiB")

        # Step5: Final BN, dropout, softmax
        start = time.time()
        out = self.hidden_dropout(self.bnw(S))
        logits = out @ self.E.weight.t()
        pred = F.softmax(logits, dim=1)
        #pred = pred.clamp(min=1e-7, max=1-1e-7)
        mem5 = torch.cuda.max_memory_allocated()/1024**2
        logging.info(f"Step5 final layers: time={time.time()-start:.4f}s, mem={mem5:.1f}MiB")

        return logits, W


# class GETD_FC(nn.Module):
#     """
#     Fully-connected Tensor-Ring GETD model with on-the-fly contraction.

#     Modes: 0 = relation, 1..k-1 = entities (k total modes).
#     Cores: one per mode, each tensor shaped by bond dimensions and a physical dimension.
#     """
#     def __init__(self, data, d_e, d_r, ni_list, rank_list, device, **kwargs): 
#         super().__init__()
#         self.k = len(ni_list)
#         assert self.k >= 2, "Need at least relation + entities"
#         assert len(rank_list) == self.k*(self.k-1)//2, \
#             f"need {self.k*(self.k-1)//2} bond ranks, got {len(rank_list)}"
#         # Precompute edges and bond ranks
#         self.edges = list(combinations(range(self.k), 2))
#         self.bond_ranks = {edge: rank_list[i] for i, edge in enumerate(self.edges)}
#         self.ni_list = ni_list
#         # Embeddings

#         self.E = torch.nn.Embedding(len(data.entities), embedding_dim=d_e, padding_idx=0)
#         self.R = torch.nn.Embedding(len(data.relations), embedding_dim=d_r, padding_idx=0)
        
#         self.E.weight.data = (1e-3 * torch.randn((len(data.entities), d_e), dtype=torch.float).to(device))
#         self.R.weight.data = (1e-3 * torch.randn((len(data.relations), d_r), dtype=torch.float).to(device))
#         # Core tensors
#         # self.cores = nn.ParameterList()
#         # for i in range(self.k):
#         #     shape = []
#         #     for j in range(self.k):
#         #         if i == j: continue
#         #         edge = (i, j) if i < j else (j, i)
#         #         shape.append(self.bond_ranks[edge])
#         #     shape.append(self.ni_list[i])
#         #     G = nn.Parameter(torch.randn(*shape, device=device) * 1e-2)
            
#         #     self.cores.append(G)
            
            
#         # Core tensors
#         self.cores = torch.nn.ParameterList()
#         for i in range(self.k):
#             shape = []
#             for j in range(self.k):
#                 if i == j:
#                     continue
#                 edge = (i, j) if i < j else (j, i)
#                 shape.append(self.bond_ranks[edge])
#             shape.append(self.ni_list[i])
#             # initialize with numpy uniform of shape tuple
#             size_tuple = tuple(shape)
#             G = torch.nn.Parameter(
#                 torch.tensor(
#                     np.random.uniform(-1e-1, 1e-1, size_tuple),
#                     dtype=torch.float,
#                     requires_grad=True
#                 ).to(device)
#             )
#             self.cores.append(G)
#         # Norms & dropout
        
#         self.bnr = nn.BatchNorm1d(d_r)
#         self.bne = nn.BatchNorm1d(d_e)
#         self.bnw = nn.BatchNorm1d(d_e)
#         self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.0))
#         self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.0))
#         self.loss = MyLoss()
        
#     def forward(self, r_idx, e_idx_list, miss, W=None):
#         B = r_idx.size(0)
#         #print(f"\n=== RUN miss={miss}  batch={B} ===")

#         # subscripts for einsum
#         bond_letters = {e: chr(ord('a')+i) for i,e in enumerate(self.edges)}
#         phys_letters = [chr(ord('A')+m) for m in range(self.k)]

#         # 1) embed relation + fuse G0
#         R0 = self.R(r_idx)
#         #print(f"[1] R0.shape = {tuple(R0.shape)}")
#         R0 = self.bnr(R0)
#         R0 = self.input_dropout(R0)
#         G0 = self.cores[0]
#         p0 = phys_letters[0]
#         b0 = [bond_letters[e] for e in self.edges if 0 in e]
#         eq0 = f"z{p0},{''.join(b0)+p0}->z{''.join(b0)}"
#         #print(f"[2] Fuse G0: G0.shape={tuple(G0.shape)}  einsum='{eq0}'")
#         T = torch.einsum(eq0, R0, G0)
#         sub = 'z' + ''.join(b0)
#         #print(f"[2] → T.shape={tuple(T.shape)}, sub='{sub}'")

#         # 2) fuse all non-missing entity cores (keep all bonds)
#         for m in range(1, self.k):
#             if m == miss: continue
#             Gm = self.cores[m]
#             pm = phys_letters[m]
#             bm = [bond_letters[e] for e in self.edges if m in e]
#             eqf = f"{sub},{''.join(bm)+pm}->{sub+pm}"
#             #print(f"[3.{m}] Fuse G{m}: shape={tuple(Gm.shape)}  einsum='{eqf}'")
#             T = torch.einsum(eqf, T, Gm)
#             sub += pm
#             #print(f"[3.{m}] → T.shape={tuple(T.shape)}, sub='{sub}'")

#         # 3) absorb all non-missing embeddings (collapse phys legs only)
#         ei = 0
#         for m in range(1, self.k):
#             if m == miss: continue
#             E_m = self.E(e_idx_list[ei]); ei += 1
#             pm = phys_letters[m]
#             eqa = f"{sub},z{pm}->{sub.replace(pm, '')}"
#             #print(f"[4.{m}] Absorb E{m}: E.shape={tuple(E_m.shape)}  einsum='{eqa}'")
#             E_m = self.bne(E_m)
#             E_m = self.input_dropout(E_m)
#             T = torch.einsum(eqa, T, E_m)
#             sub = sub.replace(pm, '')
#             #print(f"[4.{m}] → T.shape={tuple(T.shape)}, sub='{sub}'")

#         # 4) fuse missing core + collapse
#         Gm = self.cores[miss]
#         pm = phys_letters[miss]
#         bm = [bond_letters[e] for e in self.edges if miss in e]
#         eqm = f"{sub},{''.join(bm)+pm}->{sub+pm}"
#         #print(f"[5] Fuse missing G{miss}: shape={tuple(Gm.shape)}  einsum='{eqm}'")
#         T = torch.einsum(eqm, T, Gm)
#         sub += pm
#         #print(f"[5] → T.shape={tuple(T.shape)}, sub='{sub}'")

#         # collapse all bond dims
#         S = T
#         for _ in range(S.dim()-2):
#             S = S.sum(dim=1)
#         #print(f"[6] After collapse S.shape={tuple(S.shape)}")

#         # final BN+dropout+softmax
#         out = self.bnw(S)
#         out = self.hidden_dropout(out)
#         logits = out @ self.E.weight.t()
#         return F.softmax(logits, dim=1), W

#     # def forward(self, r_idx, e_idx_list, miss_ent_domain, W=None): ##doesn't save legs to missed one
#     #     """
#     #     r_idx:           (B,)   relation indices
#     #     e_idx_list:      list of (k-2) tensors each (B,) for the known entities
#     #     miss_ent_domain: int ∈ [1..k-1]
#     #     """
#     #     B = r_idx.size(0)
#     #     #print(f"\n[DEBUG] B={B}, missing mode={miss_ent_domain}")
#     #     # 1) embed relation
#     #     r = self.R(r_idx)         # (B, d_r)
#     #     #print(f"[1] r.shape = {tuple(r.shape)}")
#     #     r = self.bnr(r)
#     #     r = self.input_dropout(r)

#     #     # letter assignment for einsum
#     #     bond_letters = {e: chr(ord('a')+i) for i,e in enumerate(self.edges)}
#     #     phys_letters = [chr(ord('i')+m) for m in range(self.k)]

#     #     # 2) fuse relation into core G₀
#     #     G0 = self.cores[0]
#     #     p0    = phys_letters[0]
#     #     bonds0= [bond_letters[e] for e in self.edges if 0 in e]
#     #     eq0   = f"z{p0},{''.join(bonds0)+p0}->z{''.join(bonds0)}"
#     #     #print(f"[2] Fuse G0: G0.shape={tuple(G0.shape)}, einsum='{eq0}'")
#     #     T     = torch.einsum(eq0, r, G0)
#     #     sub_T = 'z' + ''.join(bonds0)
#     #     #print(f"[2] T.shape={tuple(T.shape)}, sub_T='{sub_T}'")
#     #     # 3) fuse & absorb all non‑missing entity cores
#     #     idx_known = 0
#     #     for mode in range(1, self.k):
#     #         if mode == miss_ent_domain:
#     #             continue

#     #         # fuse core G_mode
#     #         Gm = self.cores[mode]
#     #         pm = phys_letters[mode]
#     #         bonds_m = [bond_letters[e] for e in self.edges if mode in e]
#     #         eq_f = f"{sub_T},{''.join(bonds_m)+pm}->{sub_T+pm}"
#     #         #print(f"[3.{mode}] Fuse G{mode}: G.shape={tuple(Gm.shape)}, einsum='{eq_f}'")
#     #         T   = torch.einsum(eq_f, T, Gm)
#     #         #print(f"[3.{mode}] After fuse, T.shape={tuple(T.shape)}")
#     #         # absorb known entity embedding
#     #         e = self.E(e_idx_list[idx_known]);  idx_known += 1
#     #         e = self.bne(e)
#     #         e = self.input_dropout(e)
#     #         eq_a = f"{sub_T+pm},z{pm}->{sub_T}"
#     #         #print(f"[3.{mode}] Absorb e{mode}: e.shape={tuple(e.shape)}, einsum='{eq_a}'")
#     #         T   = torch.einsum(eq_a, T, e)
#     #         #print(f"[3.{mode}] After absorb, T.shape={tuple(T.shape)}")
#     #     # 4) now project into the missing mode
#     #     # fuse missing core
#     #     Gm = self.cores[miss_ent_domain]
#     #     pm = phys_letters[miss_ent_domain]
#     #     bonds_m = [bond_letters[e] for e in self.edges if miss_ent_domain in e]
#     #     eq_fm = f"{sub_T},{''.join(bonds_m)+pm}->{sub_T+pm}"
#     #     #print(f"[4] Fuse missing G{miss_ent_domain}: G.shape={tuple(Gm.shape)}, einsum='{eq_fm}'")
#     #     T    = torch.einsum(eq_fm, T, Gm)
#     #     #print(f"[4] After missing fuse, T.shape={tuple(T.shape)}")
#     #     # collapse all bond dims → S[b, i_miss]
#     #     S = T
#     #     for _ in range(S.dim()-2):
#     #         S = S.sum(dim=1)
#     #     #print(f"[4] After collapse, S.shape={tuple(S.shape)}")
#     #     # final prediction
#     #     out    = self.bnw(S)
#     #     out    = self.hidden_dropout(out)
#     #     logits = out @ self.E.weight.t()     # (B, n_entities)
#     #     return F.softmax(logits, dim=1), W

# class GETD_FC(nn.Module):
        
#     def __init__(self, d, d_e, d_r, k, ni_list, rank_list, device, **kwargs):
#         """
#         Fully‐connected TN with k cores (k–1 entities + 1 relation).

#         Args:
#           d           : dataset object with d.entities, d.relations
#           d_e         : entity‐embedding dim
#           d_r         : relation‐embedding dim
#           k           : total modes (here 4: three entities + one relation)
#           ni_list     : list of length k giving [n1, n2, n3, n_rel]
#           rank_list   : flat list of length k*(k-1)//2 giving R_ij for each i<j
#           device      : torch device
#         """
#         super(GETD_FC, self).__init__()
#         assert len(ni_list) == k, "ni_list must have length k"
#         assert len(rank_list) == k*(k-1)//2, f"need {k*(k-1)//2} ranks"

#         # 1) build the list of all edges (i<j) in the complete graph on k nodes
#         self.edges = list(combinations(range(k), 2))
#         self.ary = len(d.train_data[0]) - 1
#         # 2) map flat rank_list → dict {(i,j): R_ij}
#         edge_ranks = { self.edges[i]: rank_list[i]
#                        for i in range(len(self.edges)) }

#         # 3) embeddings
#         self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
#         self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
#         nn.init.normal_(self.E.weight, std=1e-3)
#         nn.init.normal_(self.R.weight, std=1e-3)

#         # 4) create one core‐tensor per mode i
#         #    each core G[i] has a bond‐leg for each j≠i of size R_{min(i,j),max(i,j)},
#         #    plus a “physical” leg of size ni_list[i].
#         self.cores = nn.ParameterList()
#         for i in range(k):
#             shape = []
            
#             for j in range(k):
#                 if j == i:
#                     continue
#                 e = (i,j) if i<j else (j,i)
#                 shape.append(edge_ranks[e])
#             shape.append(ni_list[i])
#             # append the dangling (physical) dimension
            
#             # Parameter of shape [R_{i,*}, n_i]
#             G_i = nn.Parameter(torch.randn(*shape, device=device) * 1e-1)
#             self.cores.append(G_i)

#         # 5) other modules
#         self.bnr = nn.BatchNorm1d(d_r)
#         self.bne = nn.BatchNorm1d(d_e)
#         self.bnw = nn.BatchNorm1d(d_e)
#         self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.0))
#         self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.0))
#         self.loss = MyLoss()
#         self.k = k
#         self.ni_list = ni_list
     

#     def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
#         de = self.E.weight.shape[1]
#         dr = self.R.weight.shape[1]

#         if W is None:
#             # unpack your cores
#             if self.k == 4:
#                 G0, G1, G2, G3 = self.cores
#                 # edges: (0,1)=a, (0,2)=b, (0,3)=c,
#                 #        (1,2)=d, (1,3)=e, (2,3)=f
#                 # phys dims: i,j,k,l
#                 W0 = torch.einsum(
#                     'abci,adej,bdfk,cefl->ijkl',  
#                     G0, G1, G2, G3
#                 )
#                 # W_data.shape == (n0,n1,n2,n3)

#             elif self.k == 5:
#                 G0, G1, G2, G3, G4 = self.cores
#                 # edges → letters:
#                 # (0,1)=a, (0,2)=b, (0,3)=c, (0,4)=d,
#                 # (1,2)=e, (1,3)=f, (1,4)=g,
#                 # (2,3)=h, (2,4)=i,
#                 # (3,4)=j
#                 # phys dims → p,q,r,s,t
#                 W0 = torch.einsum(
#                     'abcdp,'   # G0: a,b,c,d → phys p
#                     'aefgq,'   # G1: a,e,f,g → phys q
#                     'behir,'   # G2: b,e,h,i → phys r
#                     'cfhjs,'   # G3: c,f,h,j → phys s
#                     'dgijt->pqrst',  # G4: d,g,i,j → phys t
#                     G0, G1, G2, G3, G4
#                 )
#                 # W_data.shape == (n0,n1,n2,n3,n4)

#             else:
#                 raise ValueError(f"FC TN not implemented for k={self.k}")

            
#             if self.ary == 3:
#                 W = W0.view(dr, de, de, de)
#             elif self.ary == 4:
#                 W = W0.view(dr, de, de, de, de)

#         r = self.bnr(self.R(r_idx))
#         W_mat = torch.mm(r, W.view(r.size(1), -1))

#         if self.ary == 3:
#             W_mat = W_mat.view(-1, de, de, de)
#             e2, e3 = self.bne(self.E(e_idx[0])), self.bne(self.E(e_idx[1]))
#             e2, e3 = self.input_dropout(e2), self.input_dropout(e3)
#             if miss_ent_domain == 1:
#                 W_mat1 = torch.einsum('ijkl,il,ik->ij', W_mat, e3, e2)
#             elif miss_ent_domain == 2:
#                 W_mat1 = torch.einsum('ijkl,il,ij->ik', W_mat, e3, e2)
#             elif miss_ent_domain == 3:
#                 W_mat1 = torch.einsum('ijkl,ij,ik->il', W_mat, e2, e3)

#         elif self.ary == 4:
#             W_mat = W_mat.view(-1, de, de, de, de)
#             e2, e3, e4 = [self.bne(self.E(e_idx[i])) for i in range(3)]
#             e2, e3, e4 = [self.input_dropout(e) for e in (e2, e3, e4)]

#             if miss_ent_domain == 1:
#                 W_mat1 = torch.einsum('ijklm,il,ik,im->ij', W_mat, e3, e2, e4)
#             elif miss_ent_domain == 2:
#                 W_mat1 = torch.einsum('ijklm,il,ij,im->ik', W_mat, e3, e2, e4)
#             elif miss_ent_domain == 3:
#                 W_mat1 = torch.einsum('ijklm,ij,ik,im->il', W_mat, e2, e3, e4)
#             elif miss_ent_domain == 4:
#                 W_mat1 = torch.einsum('ijklm,ij,ik,il->im', W_mat, e2, e3, e4)

#         W_mat1 = self.bnw(W_mat1)
#         W_mat1 = self.hidden_dropout(W_mat1)
#         x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))

#         pred = F.softmax(x, dim=1)

#         return pred, W


class GETD(torch.nn.Module):
    def __init__(self, d, d_e, d_r, k, ni_list, ranks_list, device, **kwargs):
        super(GETD, self).__init__()

        assert len(ni_list) == k, "ni_list length should be equal to k"
        assert len(ranks_list) == k, "Tensor Ring requires exactly k ranks (cyclic)."
        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        
        # Customizable ni_list per dimension
        

        self.Zlist = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.tensor(
                    np.random.uniform(-1e-1, 1e-1, 
                                    (ranks_list[i], ni_list[i], ranks_list[(i+1) % k])),
                    dtype=torch.float, requires_grad=True
                ).to(device)
            ) for i in range(k)
        ])

        # self.Zlist = torch.nn.ParameterList([
        #     torch.nn.Parameter(
        #         torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks, ni_list[i], ranks)),
        #                      dtype=torch.float, requires_grad=True).to(device)
        #     ) for i in range(k)
        # ])

        self.loss = MyLoss()
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm1d(d_e)
        self.bnr = torch.nn.BatchNorm1d(d_r)
        self.bnw = torch.nn.BatchNorm1d(d_e)
        self.ary = len(d.train_data[0]) - 1

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        de = self.E.weight.shape[1]
        dr = self.R.weight.shape[1]

        if W is None:
            Zlist = [Z for Z in self.Zlist]
            k = len(Zlist)
            einsum_str = None
            
            if k == 4:
                einsum_str = 'aib,bjc,ckd,dla->ijkl'
            elif k == 5:
                einsum_str = 'aib,bjc,ckd,dle,ema->ijklm'
            else:
                raise ValueError("TR equation for k={} is not defined.".format(k))
            
            W0 = torch.einsum(einsum_str, Zlist)
            
            if self.ary == 3:
                W = W0.view(dr, de, de, de)
            elif self.ary == 4:
                W = W0.view(dr, de, de, de, de)

        r = self.bnr(self.R(r_idx))
        W_mat = torch.mm(r, W.view(r.size(1), -1))

        if self.ary == 3:
            W_mat = W_mat.view(-1, de, de, de)
            e2, e3 = self.bne(self.E(e_idx[0])), self.bne(self.E(e_idx[1]))
            e2, e3 = self.input_dropout(e2), self.input_dropout(e3)
            if miss_ent_domain == 1:
                W_mat1 = torch.einsum('ijkl,il,ik->ij', W_mat, e3, e2)
            elif miss_ent_domain == 2:
                W_mat1 = torch.einsum('ijkl,il,ij->ik', W_mat, e3, e2)
            elif miss_ent_domain == 3:
                W_mat1 = torch.einsum('ijkl,ij,ik->il', W_mat, e2, e3)

        elif self.ary == 4:
            W_mat = W_mat.view(-1, de, de, de, de)
            e2, e3, e4 = [self.bne(self.E(e_idx[i])) for i in range(3)]
            e2, e3, e4 = [self.input_dropout(e) for e in (e2, e3, e4)]

            if miss_ent_domain == 1:
                W_mat1 = torch.einsum('ijklm,il,ik,im->ij', W_mat, e3, e2, e4)
            elif miss_ent_domain == 2:
                W_mat1 = torch.einsum('ijklm,il,ij,im->ik', W_mat, e3, e2, e4)
            elif miss_ent_domain == 3:
                W_mat1 = torch.einsum('ijklm,ij,ik,im->il', W_mat, e2, e3, e4)
            elif miss_ent_domain == 4:
                W_mat1 = torch.einsum('ijklm,ij,ik,il->im', W_mat, e2, e3, e4)

        W_mat1 = self.bnw(W_mat1)
        W_mat1 = self.hidden_dropout(W_mat1)
        x = torch.mm(W_mat1, self.E.weight.transpose(1, 0))

        pred = F.softmax(x, dim=1)

        return pred, W
    
class GETD_HT2(nn.Module):
    def __init__(self, d, d_e, d_r, k, ni, r, device, **kwargs):
        super(GETD_HT2, self).__init__()
        # — entity & relation embeddings —
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        nn.init.normal_(self.E.weight,   0, 1e-3)
        nn.init.normal_(self.R.weight,   0, 1e-3)

        # — arity & HT rank —
        assert len(d.train_data[0]) - 1 == 4, "only 4-ary supported"
        self.d_e = d_e
        self.d_r = d_r
        self.ary = 4
        # let’s bump r up to give HT more capacity
        #r = ranks * 2

        # — Level-1 cores (merge entity pairs) —
        #   ht_left[a,i,j]  merges (e1,e2) → rank-dim a
        #   ht_right[b,k,l] merges (e3,e4) → rank-dim b
        self.ht_left     = nn.Parameter(torch.randn(r, d_e, d_e) * 1e-1)
        self.ht_right    = nn.Parameter(torch.randn(r, d_e, d_e) * 1e-1)

        # — Level-2 core (merge the two rank-vectors) —
        self.ht_internal = nn.Parameter(torch.randn(r, r, r) * 1e-1)

        # — Root core (produce the final [dr, de, de, de, de] weight-tensor) —
        self.ht_root     = nn.Parameter(torch.randn(d_r, r) * 1e-1)

        # — non-linear “bells & whistles” between HT levels —
        self.ln1   = nn.LayerNorm(r)
        self.dp1   = nn.Dropout(0.1)
        self.ln2   = nn.LayerNorm(r)
        self.dp2   = nn.Dropout(0.1)

        # — dropouts & batch-norm on embeddings & final scores —
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.2))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.2))
        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)
        self.bnw = nn.BatchNorm1d(d_e)

        self.loss = MyLoss()

    def build_W(self):
        """
        Build the full 5-D weight tensor W[dr, i, j, k, l] via HT:
          1) c,i,j,k,l = sum_{a,b} ht_internal[c,a,b] * ht_left[a,i,j] * ht_right[b,k,l]
          2) d,i,j,k,l = sum_c ht_root[d,c] * c,i,j,k,l
        """
        # 1) merge e1/e2 & e3/e4 → internal rank‐vector
        #    ht_internal[c,a,b], ht_left[a,i,j], ht_right[b,k,l] → W_int[c,i,j,k,l]
        W_int = torch.einsum('cab,aij,bkl->cijkl',
                             self.ht_internal,   # [r,   r,   r]
                             self.ht_left,       # [r,   de,  de]
                             self.ht_right)      # [r,   de,  de]
        # 2) merge relation‐axis
        #    ht_root[d,c], W_int[c,i,j,k,l] → W[d,i,j,k,l]
        W = torch.einsum('dc,cijkl->dijkl',
                         self.ht_root,  # [dr,  r]
                         W_int)         # [r,   de,  de,  de,  de]
        # final: [dr, de, de, de, de]
        return W

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        B  = r_idx.size(0)
        de = self.E.embedding_dim
        dr = self.R.embedding_dim

        # 1) build the weight tensor if not passed in
        if W is None:
            W = self.build_W()           # [dr, de, de, de, de]

        # 2) slice out each example’s core, conditioned on the relation embedding
        r_emb = self.bnr(self.R(r_idx))     # [B, dr]
        W_mat = torch.mm(r_emb, W.view(dr, -1))  # [B, de^4]
        W_mat = W_mat.view(B, de, de, de, de)     # [B, de, de, de, de]

        # 3) gather & normalize the three KNOWN entity embeddings
        #    e_idx is a tuple of three indices in the order needed by miss_ent_domain
        if miss_ent_domain == 1:
            e2,e3,e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
        elif miss_ent_domain == 2:
            e1,e3,e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
        elif miss_ent_domain == 3:
            e1,e2,e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
        else:  # miss_ent_domain==4
            e1,e2,e3 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]

        # 4) contract away the three known dims, leaving a score vector across the missing entity’s axis
        #    e.g. if miss_ent_domain==1 we leave the  i-th axis:
        if miss_ent_domain == 1:
            # W_mat[b,i,j,k,l], contract j→e2, k→e3, l→e4 → leave i
            W_out = torch.einsum('bijkl,bj,bk,bl->bi', W_mat, e2, e3, e4)
        elif miss_ent_domain == 2:
            W_out = torch.einsum('bijkl,bi,bk,bl->bj', W_mat, e1, e3, e4)
        elif miss_ent_domain == 3:
            W_out = torch.einsum('bijkl,bi,bj,bl->bk', W_mat, e1, e2, e4)
        else:
            W_out = torch.einsum('bijkl,bi,bj,bk->bl', W_mat, e1, e2, e3)
        # now W_out has shape [B, de]

        # 5) normalize, dropout → final scores over all entities
        W_out = self.bnw(W_out)                # [B, de]
        W_out = self.hidden_dropout(W_out)     # [B, de]
        x     = torch.mm(W_out, self.E.weight.t())  # [B, #entities]
        pred  = F.softmax(x, dim=1)
        return pred, W




class HT(nn.Module):
    def __init__(self, d, d_e, d_r, k, ni, ranks, device, **kwargs):
        super(HT, self).__init__()
        self.E = nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float, device=device))
        self.R.weight.data = (1e-3 * torch.randn((len(d.relations), d_r), dtype=torch.float, device=device))
        self.input_dropout = nn.Dropout(kwargs.get("input_dropout", 0.2))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.2))
        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)
        self.bnw = nn.BatchNorm1d(d_e)
        self.ary = len(d.train_data[0]) - 1  # should be 4 for WikiPeople-4
        self.loss = MyLoss()
        self.rank = ranks

        # HT core tensors
        self.ht_root = nn.Parameter(torch.randn(self.rank, self.rank, d_r) * 1e-1)  # (r, r, d_r)
        self.ht_left = nn.Parameter(torch.randn(self.rank, d_e, d_e) * 1e-1)        # (r, d_e, d_e)
        self.ht_right = nn.Parameter(torch.randn(self.rank, d_e, d_e) * 1e-1)       # (r, d_e, d_e)

    def batched_entity_scores(self, score_fn, B, num_entities, chunk_size=512):
        device = next(self.parameters()).device
        scores = []
        for start in range(0, num_entities, chunk_size):
            end = min(start + chunk_size, num_entities)
            cand_idx = torch.arange(start, end, device=device)
            score_chunk = score_fn(cand_idx)  # returns [B, chunk_size]
            scores.append(score_chunk)
        return torch.cat(scores, dim=1)  # [B, num_entities]

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        device = next(self.parameters()).device
        de = self.E.weight.shape[1]
        dr = self.R.weight.shape[1]
        num_entities = self.E.num_embeddings
        B = r_idx.size(0)
        rank = self.rank

        # Get relation embedding
        r = self.bnr(self.R(r_idx))  # [B, d_r]

        # Get entity embeddings for given entities
        e_emb = [self.E(e_idx[i]) for i in range(len(e_idx))]
        e_emb = [self.bne(e) for e in e_emb]
        e_emb = [self.input_dropout(e) for e in e_emb]

        # Miss entity domain: which entity to predict (1-based)
        # We'll score all possible candidates for that slot
        chunk_size = 16  # adjust based on your GPU memory

        if miss_ent_domain == 1:
            # Predicting e1, given e2, e3, e4
            e2, e3, e4 = e_emb[0], e_emb[1], e_emb[2]
            def score_fn(cand_idx):
                e1_cand = self.E(cand_idx)                     # [chunk, d_e]
                e1_cand = self.bne(e1_cand)
                e1_cand = self.input_dropout(e1_cand)
                # Expand for batch: [B, chunk, d_e]
                e1_exp = e1_cand.unsqueeze(0).expand(B, -1, de)
                e2_exp = e2.unsqueeze(1).expand(B, len(cand_idx), de)
                lvec = torch.einsum('aij,bnj,bni->bna', self.ht_left, e1_exp, e2_exp) # (B, chunk, r)
                e3_exp = e3.unsqueeze(1).expand(B, len(cand_idx), de)
                e4_exp = e4.unsqueeze(1).expand(B, len(cand_idx), de)
                rvec = torch.einsum('aij,bnj,bni->bna', self.ht_right, e3_exp, e4_exp) # (B, chunk, r)
                # Root: (r, r, d_r), lvec: (B, chunk, r), rvec: (B, chunk, r)
                s = torch.einsum('dij,bni,bnj->bnd', self.ht_root, lvec, rvec)         # (B, chunk, d_r)
                # Now contract with relation embedding r [B, d_r]
                score = torch.einsum('bd,bnd->bn', r, s)                               # (B, chunk)
                return score
            scores = self.batched_entity_scores(score_fn, B, num_entities, chunk_size)
        elif miss_ent_domain == 2:
            # Predicting e2, given e1, e3, e4
            e1, e3, e4 = e_emb[0], e_emb[1], e_emb[2]
            def score_fn(cand_idx):
                e2_cand = self.E(cand_idx)
                e2_cand = self.bne(e2_cand)
                e2_cand = self.input_dropout(e2_cand)
                e1_exp = e1.unsqueeze(1).expand(B, len(cand_idx), de)
                e2_exp = e2_cand.unsqueeze(0).expand(B, -1, de)
                lvec = torch.einsum('aij,bnj,bni->bna', self.ht_left, e1_exp, e2_exp)
                e3_exp = e3.unsqueeze(1).expand(B, len(cand_idx), de)
                e4_exp = e4.unsqueeze(1).expand(B, len(cand_idx), de)
                rvec = torch.einsum('aij,bnj,bni->bna', self.ht_right, e3_exp, e4_exp)
                s = torch.einsum('dij,bni,bnj->bnd', self.ht_root, lvec, rvec)
                score = torch.einsum('bd,bnd->bn', r, s)
                return score
            scores = self.batched_entity_scores(score_fn, B, num_entities, chunk_size)
        elif miss_ent_domain == 3:
            # Predicting e3, given e1, e2, e4
            e1, e2, e4 = e_emb[0], e_emb[1], e_emb[2]
            def score_fn(cand_idx):
                e3_cand = self.E(cand_idx)
                e3_cand = self.bne(e3_cand)
                e3_cand = self.input_dropout(e3_cand)
                e1_exp = e1.unsqueeze(1).expand(B, len(cand_idx), de)
                e2_exp = e2.unsqueeze(1).expand(B, len(cand_idx), de)
                lvec = torch.einsum('aij,bnj,bni->bna', self.ht_left, e1_exp, e2_exp)
                e3_exp = e3_cand.unsqueeze(0).expand(B, -1, de)
                e4_exp = e4.unsqueeze(1).expand(B, len(cand_idx), de)
                rvec = torch.einsum('aij,bnj,bni->bna', self.ht_right, e3_exp, e4_exp)
                s = torch.einsum('dij,bni,bnj->bnd', self.ht_root, lvec, rvec)
                score = torch.einsum('bd,bnd->bn', r, s)
                return score
            scores = self.batched_entity_scores(score_fn, B, num_entities, chunk_size)
        elif miss_ent_domain == 4:
            # Predicting e4, given e1, e2, e3
            e1, e2, e3 = e_emb[0], e_emb[1], e_emb[2]
            def score_fn(cand_idx):
                e4_cand = self.E(cand_idx)
                e4_cand = self.bne(e4_cand)
                e4_cand = self.input_dropout(e4_cand)
                e1_exp = e1.unsqueeze(1).expand(B, len(cand_idx), de)
                e2_exp = e2.unsqueeze(1).expand(B, len(cand_idx), de)
                lvec = torch.einsum('aij,bnj,bni->bna', self.ht_left, e1_exp, e2_exp)
                e3_exp = e3.unsqueeze(1).expand(B, len(cand_idx), de)
                e4_exp = e4_cand.unsqueeze(0).expand(B, -1, de)
                rvec = torch.einsum('aij,bnj,bni->bna', self.ht_right, e3_exp, e4_exp)
                s = torch.einsum('dij,bni,bnj->bnd', self.ht_root, lvec, rvec)
                score = torch.einsum('bd,bnd->bn', r, s)
                return score
            scores = self.batched_entity_scores(score_fn, B, num_entities, chunk_size)
        else:
            raise ValueError(f"miss_ent_domain {miss_ent_domain} not supported")

        # BatchNorm, Dropout, and Softmax over entity candidates
        #scores = self.bnw(scores)
        scores = self.hidden_dropout(scores)
        pred = F.softmax(scores, dim=1)
        return pred, None




class GETD_TN(nn.Module):
    def __init__(self, d, d_e, d_r, tensor_kind='TR', ranks=40, k=4, device='cuda:0', **kwargs):
        super(GETD_TN, self).__init__()
        self.tensor_kind = tensor_kind.upper()
        self.d_e, self.d_r = d_e, d_r
        self.device = device
        self.ary = len(d.train_data[0]) - 1

        # Embeddings
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        self.E.weight.data.normal_(0, 1e-3)
        self.R.weight.data.normal_(0, 1e-3)

        # Base tensor on CPU
        tensor_shape = [d_r] + [d_e] * self.ary
        base_tensor = torch.randn(tensor_shape)

        # Decomposition
        if self.tensor_kind == "TUCKER":
            core, factors = tucker(
                base_tensor,
                rank=[ranks] * len(tensor_shape)
            )
            self.core = nn.Parameter(core.to(device))
            self.factors = nn.ParameterList([
                nn.Parameter(f.to(device)) for f in factors
            ])
        elif self.tensor_kind == "TT":
            factors = tensor_train(
                base_tensor,
                rank=ranks
            )
            self.core = None
            self.factors = nn.ParameterList([
                nn.Parameter(f.to(device)) for f in factors
            ])
        elif self.tensor_kind == "TR":
            factors = tensor_ring(
                base_tensor,
                rank=ranks
            )
            self.core = None
            self.factors = nn.ParameterList([
                nn.Parameter(f.to(device)) for f in factors
            ])
        else:
            raise ValueError(f"Unsupported tensor network type: {tensor_kind}")

        # Dropout & BatchNorm
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.3))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.3))
        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)
        self.bnw = nn.BatchNorm1d(d_e)
        self.loss = MyLoss()

    def reconstruct_tensor(self):
        if self.tensor_kind == "TUCKER":
            return tl.tucker_to_tensor((self.core, list(self.factors)))
        elif self.tensor_kind == "TT":
            return tl.tt_to_tensor(list(self.factors))
        elif self.tensor_kind == "TR":
            return tl.tr_to_tensor(list(self.factors))

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        # 1) Reconstruct or reuse tensor
        if W is None:
            W = self.reconstruct_tensor().to(self.device)
        else:
            W = W.to(self.device)

        # 2) Project relation embeddings
        r = self.bnr(self.R(r_idx))  # (B, d_r)
        W_flat = W.reshape(self.d_r, -1)  # (d_r, prod(d_e))
        W_mat  = torch.mm(r, W_flat)      # (B, prod(d_e))
        W_mat  = W_mat.reshape(-1, *([self.d_e] * self.ary))  # (B, d_e, ...)

        # 3) Entity embeddings
        e_embs = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]

        # 4) Build einsum equation
        labs     = [chr(ord('i') + i) for i in range(self.ary)]
        in_modes = "".join(labs)
        out_lab  = labs[miss_ent_domain - 1]
        eq = f"b{in_modes}," + ",".join(f"b{l}" for l in in_modes if l != out_lab) + f"->b{out_lab}"

        # 5) Contract
        W_out = torch.einsum(eq, W_mat, *e_embs)
        W_out = self.hidden_dropout(self.bnw(W_out))

        # 6) Final scores
        scores = torch.mm(W_out, self.E.weight.T)
        pred   = F.softmax(scores, dim=1)
        return pred, W