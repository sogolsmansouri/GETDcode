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


class GETD_HT3_Enhanced(nn.Module):
    """
    Hierarchical‐Tucker (“HT‐NI”) with:
      1) Separate HT ranks (r1,r2,r3) instead of forcing r_i = n_i.
      2) A small two‐layer MLP (with LayerNorm & residual) after the HT contraction,
         prior to scoring against all entity embeddings.
      3) Supports both arity=3 and arity=4.

    Constructor args:
      d         : dataset object, where each d.train_data[i] has length = arity+1
      d_e       : base entity embedding dim (we will project each entity into its n_i)
      d_r       : base relation embedding dim (we will project relation into its n_rel)
      k         : number of modes = (arity + 1)
      ni_list   : list of k positive ints = [n1, n2, ... , n_k]
      ht_ranks  : list of 3 positive ints = [r1, r2, r3]
      device    : "cuda" or "cpu"
      kwargs    : may contain "input_dropout", "hidden_dropout", "mlp_hidden"
                  where mlp_hidden is the hidden size of the small MLP (default = max(n_i)//2)
    """
    def __init__(self, d, d_e, d_r, k, ni_list, ht_ranks, device, **kwargs):
        ht_tree = kwargs.pop("ht_tree", "A")
        ht4_tree = kwargs.pop("ht4_tree", "A")
        super(GETD_HT3_Enhanced, self).__init__()
        # — Basic checks —
        assert len(ni_list) == k, "ni_list length must equal k"
        self.ary = len(d.train_data[0]) - 1
        assert self.ary in (3,4), "Supports only arity=3 or arity=4"
        
        if self.ary == 3:
            assert ht_tree in ("A", "C"), "ht_tree must be 'A' or 'C' for arity=3"
            self.ht_tree = ht_tree
        else:
            self.ht_tree = None
            self.ht4_tree = ht4_tree
            
        # — Save dims —
        self.d_e = d_e
        self.d_r = d_r
        self.device = device

        # — Shared embeddings (raw) —
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        nn.init.normal_(self.E.weight, 0.0, 1e-3)
        nn.init.normal_(self.R.weight, 0.0, 1e-3)

        # — Dropout & BatchNorm on raw embeddings —
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        # self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.2))
        # self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.2))
        # batchnorm on HT-MLP output; will initialize once we know n_missing
        self.bnw = None

        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)

        self.loss = MyLoss()

        # — Extract ni_list —
        if self.ary == 4:
            # 5 mode sizes: n1,n2,n3,n4,n5
            n1, n2, n3, n4, n5 = ni_list
        else:
            # 4 mode sizes: n1,n2,n3,n4
            n1, n2, n3, n4 = ni_list

        # — Extract HT ranks (r1,r2,r3) —
        assert isinstance(ht_ranks, (list,tuple)) and len(ht_ranks) == 3
        r1, r2, r3 = ht_ranks
        assert r1>0 and r2>0 and r3>0, "HT ranks must be positive"
        self.r1, self.r2, self.r3 = r1, r2, r3

        # — Build the small projection layers for each slot —
        if self.ary == 4:
            # 4 entity slots + 1 relation = 5 modes
            self.e1_proj  = nn.Linear(d_e, n1, bias=False)
            self.e2_proj  = nn.Linear(d_e, n2, bias=False)
            self.e3_proj  = nn.Linear(d_e, n3, bias=False)
            self.e4_proj  = nn.Linear(d_e, n4, bias=False)
            self.rel_proj = nn.Linear(d_r, n5, bias=False)

            # Initialize projection weights
            for lin in (self.e1_proj, self.e2_proj, self.e3_proj, self.e4_proj, self.rel_proj):
                nn.init.xavier_uniform_(lin.weight)

            # # — HT cores (arity=4), on CPU initially —
            # self.ht_left     = nn.Parameter(torch.randn(r1,   n1,    n2) * 1e-1)
            # self.ht_right    = nn.Parameter(torch.randn(r2,   n3,    n4) * 1e-1)
            # self.ht_internal = nn.Parameter(torch.randn(r3,   r1,    r2) * 1e-1)
            # self.ht_root     = nn.Parameter(torch.randn(n5,   r3) * 1e-1)
            # ――― MODIFIED: 4-ary HT cores depend on ht4_tree ―――
            if self.ht4_tree == "A":
                # Tree A (default): (e1,e2)->r1, (e3,e4)->r2, (r1,r2)->r3 -> n5
                self.ht_left     = nn.Parameter(torch.randn(r1, n1, n2) * 1e-1)
                self.ht_right    = nn.Parameter(torch.randn(r2, n3, n4) * 1e-1)
                self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2) * 1e-1)
                self.ht_root     = nn.Parameter(torch.randn(n5, r3) * 1e-1)
            elif self.ht4_tree == "C":
                self.ht_left     = nn.Parameter(torch.randn(r1, n1, n4)*1e-1)  # (e1,e4)
                self.ht_right    = nn.Parameter(torch.randn(r2, n2, n3)*1e-1)  # (e2,e3)
                self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2)*1e-1)
                self.ht_root     = nn.Parameter(torch.randn(n5, r3)*1e-1)

            else:
                # Tree B: (e1,e3)->r1, (e2,e4)->r2, (r1,r2)->r3 -> n5
                self.ht_left     = nn.Parameter(torch.randn(r1, n1, n3) * 1e-1)
                self.ht_right    = nn.Parameter(torch.randn(r2, n2, n4) * 1e-1)
                self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2) * 1e-1)
                self.ht_root     = nn.Parameter(torch.randn(n5, r3) * 1e-1)

            # Store mode‐sizes
            self.n1, self.n2, self.n3, self.n4, self.n5 = n1, n2, n3, n4, n5

        else:
            # arity=3: 3 entity slots + 1 relation = 4 modes
            # self.e1_proj  = nn.Linear(d_e, n1, bias=False)
            # self.e2_proj  = nn.Linear(d_e, n2, bias=False)
            # self.e3_proj  = nn.Linear(d_e, n3, bias=False)
            # self.rel_proj = nn.Linear(d_r, n4, bias=False)
            # for lin in (self.e1_proj, self.e2_proj, self.e3_proj, self.rel_proj):
            #     nn.init.xavier_uniform_(lin.weight)

            # self.ht_left     = nn.Parameter(torch.randn(r1,   n1,    n2) * 1e-1)
            # self.ht_right    = nn.Parameter(torch.randn(r2,   n3,    n4) * 1e-1)
            # self.ht_internal = nn.Parameter(torch.randn(r3,   r1,    r2) * 1e-1)

            # self.n1, self.n2, self.n3, self.n4 = n1, n2, n3, n4
            # arity=3: 3 entity slots + 1 relation = 4 modes
            self.e1_proj  = nn.Linear(d_e, n1, bias=False)
            self.e2_proj  = nn.Linear(d_e, n2, bias=False)
            self.e3_proj  = nn.Linear(d_e, n3, bias=False)
            self.rel_proj = nn.Linear(d_r, n4, bias=False)
            for lin in (self.e1_proj, self.e2_proj, self.e3_proj, self.rel_proj):
                nn.init.xavier_uniform_(lin.weight)

            if ht_tree == "A":
                # Tree A: (e1,e2)->r1, (e3,rel)->r2, (r1,r2)->r3
                self.ht_left     = nn.Parameter(torch.randn(r1, n1, n2) * 1e-1)
                self.ht_right    = nn.Parameter(torch.randn(r2, n3, n4) * 1e-1)
                self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2) * 1e-1)
                self.ht_final_root = None  # no additional layer needed
            else:
                # Tree C with true three‐rank structure:
                self.ht_mid   = nn.Parameter(torch.randn(r1, n2, n3) * 1e-1)
                self.ht_left  = nn.Parameter(torch.randn(r2, r1, n1) * 1e-1)
                self.ht_root  = nn.Parameter(torch.randn(r3, r2) * 1e-1)
                self.ht_final = nn.Parameter(torch.randn(n4, r3) * 1e-1)

            self.n1, self.n2, self.n3, self.n4 = n1, n2, n3, n4
        # # — Placeholders for the small MLP layers; instantiate in forward() once n_missing is known —
        
        self.back_to_de = None
        self.bn_de     = None
        self.bn_missing = None
        # in __init__(), add placeholders:
        self.back_to_de_1 = None
        self.bn_de_1      = None
        self.back_to_de_2 = None
        self.bn_de_2      = None
        self.back_to_de_3 = None
        self.bn_de_3      = None

        # self.mlp_fc1 = None
        # self.mlp_ln1 = None
        # self.mlp_fc2 = None
        # self.mlp_ln2 = None
        if isinstance(device, torch.device) and device.type == "cuda":
            self.to(device)


    def build_W(self):
        """
        Build the HT core tensor W.

        Arity=4 → W shape = [n5, n1, n2, n3, n4].
        Arity=3 → W shape = [n4, n1, n2, n3].
        """
        if self.ary == 4:
            # # Merge ht_left & ht_right via ht_internal → W_int[c,i,j,k,l]
            # W_int = torch.einsum(
            #     'cab, aij, bkl -> cijkl',
            #     self.ht_internal,  # [r3, r1, r2]
            #     self.ht_left,      # [r1, n1, n2]
            #     self.ht_right      # [r2, n3, n4]
            # )
            if self.ht4_tree == "A":
                # (e1,e2)->r1, (e3,e4)->r2
                # ht_left:  [r1,n1,n2]
                # ht_right: [r2,n3,n4]
                # ht_internal: [r3,r1,r2]
                W_int = torch.einsum(
                    'cab,aij,bkl->cijkl',
                    self.ht_internal,
                    self.ht_left,
                    self.ht_right
                )
            elif self.ht4_tree == "C":
                W_int = torch.einsum('cab, ail, bjk -> cijkl',
                                    self.ht_internal,
                                    self.ht_left,    # [r1,n1,n4]
                                    self.ht_right)   # [r2,n2,n3]

            else:
                # (e1,e3)->r1, (e2,e4)->r2
                # ht_left:  [r1,n1,n3]
                # ht_right: [r2,n2,n4]
                # ht_internal: [r3,r1,r2]
                W_int = torch.einsum(
                    'cab,aik,bjl->cijkl',
                    self.ht_internal,
                    self.ht_left,
                    self.ht_right
                )
            r3, n1, n2, n3, n4 = W_int.shape

            # Flatten → [r3, (n1·n2·n3·n4)]
            W_int_flat = W_int.contiguous().view(r3, n1 * n2 * n3 * n4)

            # Multiply by ht_root [n5, r3] → [n5, (n1·n2·n3·n4)]
            W_flat = torch.mm(self.ht_root, W_int_flat)

            # Reshape → [n5, n1, n2, n3, n4]
            W = W_flat.view(self.n5, n1, n2, n3, n4)
            return W

        else:
            # # arity=3:
            # W_int = torch.einsum(
            #     'cab, aij, bkl -> cijkl',
            #     self.ht_internal,  # [r3, r1, r2]
            #     self.ht_left,      # [r1, n1, n2]
            #     self.ht_right      # [r2, n3, n4]
            # )
            # # W_int: [r3, n1, n2, n3, n4]

            # # Sum over c and permute ℓ to front → [n4, n1, n2, n3]
            # W = W_int.sum(dim=0).permute(3, 0, 1, 2).contiguous()
            # return W
            if self.ht_tree == "A":
                # — Tree A: exactly as before —
                W_int = torch.einsum(
                    'cab, aij, bkl -> cijkl',
                    self.ht_internal,  # [r3, r1, r2]
                    self.ht_left,      # [r1, n1, n2]
                    self.ht_right      # [r2, n3, n4]
                )
                # W_int: [r3, n1, n2, n3, n4]
                # Sum over c and permute ℓ (rel index) to front → [n4, n1, n2, n3]
                W = W_int.sum(dim=0).permute(3, 0, 1, 2).contiguous()
                return W

            else:  # self.ht_tree == "C"
                
                # 1) Merge (e2,e3)->r1 and (r1,e1)->r2 in one einsum:
                #    T_int[b, i1, i2, i3] = Σ_{a} ht_left[b,a,i1] * ht_mid[a,i2,i3]
                T_int = torch.einsum('bak, aij -> bkij',
                     self.ht_left,  # [r2, r1, n1]
                     self.ht_mid)   # [r1, n2, n3]
                    # Now T_int.shape == (r2, n1, n2, n3) as intended.

                # Now T_int: [r2, n1, n2, n3]
                # 2) Merge T_int [r2, *] → [r3, *] via ht_root [r3, r2]:
                T_flat = T_int.reshape(self.r2, -1)            # [r2, n1·n2·n3]
                U      = torch.mm(self.ht_root, T_flat)       # [r3, n1·n2·n3]
                # 3) Merge U [r3, *] → [n4, *] via ht_final [n4, r3]:
                W_flat = torch.mm(self.ht_final, U)           # [n4, n1·n2·n3]
                W      = W_flat.view(self.n4, self.n1, self.n2, self.n3)
                return W

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        """
        r_idx          : [B] relation indices
        e_idx          : tuple of (arity) LongTensors, each [B], indices of known entity slots
        miss_ent_domain: int ∈ {1..arity}: which entity slot is missing
        W              : optionally precomputed HT core; if None, call build_W()

        Returns:
          pred: [B, #entities]  (softmax over all entity IDs)
          W   : the HT core just built
        """
        B = r_idx.size(0)

        # 1) Build (or reuse) HT core
        if W is None:
            W = self.build_W()
            # If arity=4 → W ∈ [n5, n1, n2, n3, n4]
            # If arity=3 → W ∈ [n4, n1, n2, n3]

        # 2) “Slice by relation”:
        #    Build rel_mode ∈ [B, n_rel]
        r_raw  = self.bnr(self.R(r_idx))        # [B, d_r]
        r_dp   = self.input_dropout(r_raw)       # [B, d_r]
        r_mode = self.rel_proj(r_dp)             # [B, n_rel]

        if self.ary == 4:
            # W: [n5, n1, n2, n3, n4], flatten last 4 dims → [n5, M]
            n5 = W.shape[0]
            M  = self.n1 * self.n2 * self.n3 * self.n4
            W_flat = W.view(n5, M)               # [n5, M]

            # W_mat: [B, n1, n2, n3, n4] = r_mode [B,n5] @ W_flat [n5,M] → reshape
            W_mat = torch.mm(r_mode, W_flat).view(B, self.n1, self.n2, self.n3, self.n4)

        else:
            # arity=3: W: [n4, n1, n2, n3] → flatten last 3 dims → [n4, M]
            n4 = W.shape[0]
            M  = self.n1 * self.n2 * self.n3
            W_flat = W.view(n4, M)                # [n4, M]

            W_mat = torch.mm(r_mode, W_flat).view(B, self.n1, self.n2, self.n3)

        # 3) Pull out known entities, project, batchnorm, dropout, then contract
        if self.ary == 4:
            if miss_ent_domain == 1:
                e2_idx, e3_idx, e4_idx = e_idx
                e2_raw = self.bne(self.E(e2_idx));  e2_dp = self.input_dropout(e2_raw);  e2_mode = self.e2_proj(e2_dp)  # [B, n2]
                e3_raw = self.bne(self.E(e3_idx));  e3_dp = self.input_dropout(e3_raw);  e3_mode = self.e3_proj(e3_dp)  # [B, n3]
                e4_raw = self.bne(self.E(e4_idx));  e4_dp = self.input_dropout(e4_raw);  e4_mode = self.e4_proj(e4_dp)  # [B, n4]

                # leave mode1 (n1):
                # W_out[b,i] = sum_{j,k,l} W_mat[b,i,j,k,l] * e2_mode[b,j] * e3_mode[b,k] * e4_mode[b,l]
                W_out = torch.einsum(
                    'bijkl, bj, bk, bl -> bi',
                    W_mat, e2_mode, e3_mode, e4_mode
                )  # [B, n1]
                n_missing = self.n1

            elif miss_ent_domain == 2:
                e1_idx, e3_idx, e4_idx = e_idx
                e1_raw = self.bne(self.E(e1_idx));  e1_dp = self.input_dropout(e1_raw);  e1_mode = self.e1_proj(e1_dp)  # [B, n1]
                e3_raw = self.bne(self.E(e3_idx));  e3_dp = self.input_dropout(e3_raw);  e3_mode = self.e3_proj(e3_dp)  # [B, n3]
                e4_raw = self.bne(self.E(e4_idx));  e4_dp = self.input_dropout(e4_raw);  e4_mode = self.e4_proj(e4_dp)  # [B, n4]

                # leave mode2 (n2):
                # W_out[b,j] = sum_{i,k,l} W_mat[b,i,j,k,l] * e1_mode[b,i] * e3_mode[b,k] * e4_mode[b,l]
                W_out = torch.einsum(
                    'bijkl, bi, bk, bl -> bj',
                    W_mat, e1_mode, e3_mode, e4_mode
                )  # [B, n2]
                n_missing = self.n2

            elif miss_ent_domain == 3:
                e1_idx, e2_idx, e4_idx = e_idx
                e1_raw = self.bne(self.E(e1_idx));  e1_dp = self.input_dropout(e1_raw);  e1_mode = self.e1_proj(e1_dp)  # [B, n1]
                e2_raw = self.bne(self.E(e2_idx));  e2_dp = self.input_dropout(e2_raw);  e2_mode = self.e2_proj(e2_dp)  # [B, n2]
                e4_raw = self.bne(self.E(e4_idx));  e4_dp = self.input_dropout(e4_raw);  e4_mode = self.e4_proj(e4_dp)  # [B, n4]

                # leave mode3 (n3):
                # W_out[b,k] = sum_{i,j,l} W_mat[b,i,j,k,l] * e1_mode[b,i] * e2_mode[b,j] * e4_mode[b,l]
                W_out = torch.einsum(
                    'bijkl, bi, bj, bl -> bk',
                    W_mat, e1_mode, e2_mode, e4_mode
                )  # [B, n3]
                n_missing = self.n3

            else:  # miss_ent_domain == 4
                e1_idx, e2_idx, e3_idx = e_idx
                e1_raw = self.bne(self.E(e1_idx));  e1_dp = self.input_dropout(e1_raw);  e1_mode = self.e1_proj(e1_dp)  # [B, n1]
                e2_raw = self.bne(self.E(e2_idx));  e2_dp = self.input_dropout(e2_raw);  e2_mode = self.e2_proj(e2_dp)  # [B, n2]
                e3_raw = self.bne(self.E(e3_idx));  e3_dp = self.input_dropout(e3_raw);  e3_mode = self.e3_proj(e3_dp)  # [B, n3]

                # leave mode4 (n4):
                # W_out[b,l] = sum_{i,j,k} W_mat[b,i,j,k,l] * e1_mode[b,i] * e2_mode[b,j] * e3_mode[b,k]
                W_out = torch.einsum(
                    'bijkl, bi, bj, bk -> bl',
                    W_mat, e1_mode, e2_mode, e3_mode
                )  # [B, n4]
                n_missing = self.n4

        else:
            # ARITY = 3
            if miss_ent_domain == 1:
                e2_idx, e3_idx = e_idx
                e2_raw = self.bne(self.E(e2_idx));  e2_dp = self.input_dropout(e2_raw);  e2_mode = self.e2_proj(e2_dp)  # [B, n2]
                e3_raw = self.bne(self.E(e3_idx));  e3_dp = self.input_dropout(e3_raw);  e3_mode = self.e3_proj(e3_dp)  # [B, n3]

                # leave mode1 (n1):
                # W_out[b,i] = sum_{j,k} W_mat[b,i,j,k] * e2_mode[b,j] * e3_mode[b,k]
                W_out = torch.einsum(
                    'bijk, bj, bk -> bi',
                    W_mat, e2_mode, e3_mode
                )  # [B, n1]
                n_missing = self.n1

            elif miss_ent_domain == 2:
                e1_idx, e3_idx = e_idx
                e1_raw = self.bne(self.E(e1_idx));  e1_dp = self.input_dropout(e1_raw);  e1_mode = self.e1_proj(e1_dp)  # [B, n1]
                e3_raw = self.bne(self.E(e3_idx));  e3_dp = self.input_dropout(e3_raw);  e3_mode = self.e3_proj(e3_dp)  # [B, n3]

                # leave mode2 (n2):
                # W_out[b,j] = sum_{i,k} W_mat[b,i,j,k] * e1_mode[b,i] * e3_mode[b,k]
                W_out = torch.einsum(
                    'bijk, bi, bk -> bj',
                    W_mat, e1_mode, e3_mode
                )  # [B, n2]
                n_missing = self.n2

            else:  # miss_ent_domain == 3
                e1_idx, e2_idx = e_idx
                e1_raw = self.bne(self.E(e1_idx));  e1_dp = self.input_dropout(e1_raw);  e1_mode = self.e1_proj(e1_dp)  # [B, n1]
                e2_raw = self.bne(self.E(e2_idx));  e2_dp = self.input_dropout(e2_raw);  e2_mode = self.e2_proj(e2_dp)  # [B, n2]

                # leave mode3 (n3):
                # W_out[b,k] = sum_{i,j} W_mat[b,i,j,k] * e1_mode[b,i] * e2_mode[b,j]
                W_out = torch.einsum(
                    'bijk, bi, bj -> bk',
                    W_mat, e1_mode, e2_mode
                )  # [B, n3]
                n_missing = self.n3

        # # At this point: W_out.shape = [B, n_missing]

        # # 4) Build or reuse the two‐layer MLP now that we know n_missing
        # if self.mlp_fc1 is None or self.mlp_fc1.in_features != n_missing:
        #     mlp_hidden = max(n_missing // 2, 1)
        #     self.mlp_fc1 = nn.Linear(n_missing, mlp_hidden, bias=False).to(self.device)
        #     self.mlp_ln1 = nn.LayerNorm(mlp_hidden).to(self.device)
        #     self.mlp_fc2 = nn.Linear(mlp_hidden, n_missing, bias=False).to(self.device)
        #     self.mlp_ln2 = nn.LayerNorm(n_missing).to(self.device)
        #     nn.init.xavier_uniform_(self.mlp_fc1.weight)
        #     nn.init.xavier_uniform_(self.mlp_fc2.weight)

        #     # Now that we know n_missing, set batchnorm for HT output
        #     self.bnw = nn.BatchNorm1d(n_missing).to(self.device)
        #     self.bn_missing = nn.BatchNorm1d(n_missing).to(self.device)

        # # a) First MLP layer → [B, mlp_hidden]
        # h1 = self.mlp_fc1(W_out)        # [B, mlp_hidden]
        # h1 = self.mlp_ln1(h1)
        # h1 = F.gelu(h1)
        # h1 = self.hidden_dropout(h1)

        # # b) Second MLP layer → [B, n_missing]
        # h2 = self.mlp_fc2(h1)           # [B, n_missing]
        # h2 = self.mlp_ln2(h2)

        # # c) Residual
        # W_out_enh = W_out + h2          # [B, n_missing]

        # # 5) BatchNorm + Dropout on the enhanced output
        # W_bn = self.bnw(W_out_enh)      # [B, n_missing]
        # W_dp = self.hidden_dropout(W_bn)
        # W_bn_missing = self.bn_missing(W_out_enh)   # BatchNorm1d(n_missing)
        # W_dp_missing = self.hidden_dropout(W_bn_missing)  # [B, n_missing]
        # # # 6) Score against all entity embeddings
        # # x = torch.mm(W_dp, self.E.weight.t())  # [B, #entities]
        # # pred = F.softmax(x, dim=1)
        # # — ensure back_to_de/bn_de have the right input size:
        # if self.back_to_de is None or self.back_to_de.in_features != n_missing:
        #     # rebuild both layers now that we know n_missing
        #     self.back_to_de = nn.Linear(n_missing, self.d_e, bias=False).to(self.device)
        #     nn.init.xavier_uniform_(self.back_to_de.weight)
        #     self.bn_de = nn.BatchNorm1d(self.d_e).to(self.device)
        # # 6a) “Back-project” from [B × n_missing] down to [B × d_e]
        # W_proj = self.back_to_de(W_dp)            # [B, d_e]

        # # 6b) (optionally) batchnorm+dropout on that projected vector
        # #W_bn2  = self.bnw(W_proj)                 # [B, d_e]
        # W_bn2 = self.bn_de(W_proj)  # this BN expects exactly d_e features
        # W_dp2  = self.hidden_dropout(W_bn2)       # [B, d_e]

        # # 6c) Now we can safely multiply by E.weight.t(), since both dims match
        # x     = torch.mm(W_dp2, self.E.weight.t())  # [B, #entities]
        # pred  = F.softmax(x, dim=1)
        # return pred, W
        # if self.back_to_de is None or self.back_to_de.in_features != n_missing:
        #     # rebuild back_to_de so it maps n_missing → d_e
        #     self.back_to_de = nn.Linear(n_missing, self.d_e, bias=False).to(self.device)
        #     nn.init.xavier_uniform_(self.back_to_de.weight)
        #     self.bn_de = nn.BatchNorm1d(self.d_e).to(self.device)
        
        # W_proj = self.back_to_de(W_out)       # [B, d_e]
        # W_proj_bn = self.bn_de(W_proj)        # [B, d_e]
        # W_dp2 = self.hidden_dropout(W_proj_bn)
        # x     = torch.mm(W_dp2, self.E.weight.t())
        # pred  = F.softmax(x, dim=1)
        # return pred, W
                # … after computing W_out and n_missing …

        # 4) Back‐project via the correct branch (1, 2, or 3):
        if miss_ent_domain == 1:
            if self.back_to_de_1 is None:
                self.back_to_de_1 = nn.Linear(self.n1, self.d_e, bias=False).to(self.device)
                nn.init.xavier_uniform_(self.back_to_de_1.weight)
                self.bn_de_1 = nn.BatchNorm1d(self.d_e).to(self.device)
            W_proj = self.back_to_de_1(W_out)   # [B, d_e]
            W_proj = self.bn_de_1(W_proj)       # [B, d_e]
            W_dp2  = self.hidden_dropout(W_proj)

        elif miss_ent_domain == 2:
            if self.back_to_de_2 is None:
                self.back_to_de_2 = nn.Linear(self.n2, self.d_e, bias=False).to(self.device)
                nn.init.xavier_uniform_(self.back_to_de_2.weight)
                self.bn_de_2 = nn.BatchNorm1d(self.d_e).to(self.device)
            W_proj = self.back_to_de_2(W_out)   # [B, d_e]
            W_proj = self.bn_de_2(W_proj)       # [B, d_e]
            W_dp2  = self.hidden_dropout(W_proj)

        else:  # miss_ent_domain == 3
            if self.back_to_de_3 is None:
                self.back_to_de_3 = nn.Linear(self.n3, self.d_e, bias=False).to(self.device)
                nn.init.xavier_uniform_(self.back_to_de_3.weight)
                self.bn_de_3 = nn.BatchNorm1d(self.d_e).to(self.device)
            W_proj = self.back_to_de_3(W_out)   # [B, d_e]
            W_proj = self.bn_de_3(W_proj)       # [B, d_e]
            W_dp2  = self.hidden_dropout(W_proj)

        # 5) Final scoring:
        x    = torch.mm(W_dp2, self.E.weight.t())  # [B, #entities]
        pred = F.softmax(x, dim=1)
        return pred, W


class GETD_HT3(nn.Module):
    def __init__(self, d, d_e, d_r, k, ni_list, ranks, device, **kwargs):
        """
        d         : dataset object (each row in d.train_data has arity+1 = 4 or 5 entries)
        d_e       : entity embedding dimension
        d_r       : relation embedding dimension
        k         : number of cores (ignored except for asserting len(ni_list) = 3)
        ni_list   : list of 3 positive ints [r1, r2, r3], the HT ranks at each level
        ranks     : unused (kept for TR compatibility)
        device    : 'cuda' or 'cpu'
        kwargs    : e.g. input_dropout, hidden_dropout
        """
        super(GETD_HT3, self).__init__()

        # — entity & relation embeddings —
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        nn.init.normal_(self.E.weight, 0, 1e-3)
        nn.init.normal_(self.R.weight, 0, 1e-3)

        # — dropout & batch-norm for embeddings —
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.2))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.2))
        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)
        self.bnw = nn.BatchNorm1d(d_e)

        self.loss = MyLoss()

        # — detect arity (3 or 4) —
        self.ary = len(d.train_data[0]) - 1
        assert self.ary in (3, 4), "Only arity 3 or 4 are supported"
        self.d_e = d_e
        self.d_r = d_r

        # — parse ni_list = [r₁, r₂, r₃] —
        assert isinstance(ni_list, (list, tuple)) and len(ni_list) == 3, \
            "ni_list must be a list of 3 positive ints ([r1,r2,r3])"
        r1, r2, r3 = ni_list
        assert all(isinstance(x, int) and x > 0 for x in (r1, r2, r3)), \
            "All entries in ni_list must be positive integers"

        if self.ary == 4:
            # — arity=4 HT tree (4 entities + 1 relation) —
            #   Level-1: merge (e1,e2)→rank-r1  and  (e3,e4)→rank-r2
            self.ht_left     = nn.Parameter(torch.randn(r1, d_e, d_e, device=device) * 1e-1)
            self.ht_right    = nn.Parameter(torch.randn(r2, d_e, d_e, device=device) * 1e-1)
            #   Level-2: merge those two rank‐vectors → internal rank-r3
            self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2, device=device) * 1e-1)
            #   Root: merge internal rank‐vector (r3) with relation dim d_r
            self.ht_root     = nn.Parameter(torch.randn(d_r, r3, device=device) * 1e-1)

        else:  # self.ary == 3
            # — arity=3 HT tree (3 entities + 1 relation) —
            #   Level-1a: merge (e1,e2)→rank-r1
            self.ht_left     = nn.Parameter(torch.randn(r1, d_e, d_e, device=device) * 1e-1)
            #   Level-1b: merge (e3,relation)→rank-r2  (note: relation dimension is d_r)
            self.ht_right    = nn.Parameter(torch.randn(r2, d_e, d_r, device=device) * 1e-1)
            #   Level-2: merge those two rank‐vectors → internal rank-r3
            self.ht_internal = nn.Parameter(torch.randn(r3, r1, r2, device=device) * 1e-1)
            #   No separate “ht_root” here, since relation is already merged in ht_right.

    def build_W(self):
        """
        Build the HT core depending on arity:

        For arity=4:
          1) W_int[c,i,j,k,l] = ∑_{a,b}
                 ht_internal[c,a,b] · ht_left[a,i,j] · ht_right[b,k,l]
             → shape [r₃, d_e, d_e, d_e, d_e]
          2) flatten W_int→[r₃, d_e⁴], mm with ht_root [d_r, r₃] →[d_r, d_e⁴], reshape→[d_r, d_e, d_e, d_e, d_e]

        For arity=3:
          1) W_int[c,i,j,k,ℓ] = ∑_{a,b}
                 ht_internal[c,a,b] · ht_left[a,i,j] · ht_right[b,k,ℓ]
             where i,j=entity dims, k=third‐entity dim, ℓ=relation dim
             → shape [r₃, d_e, d_e, d_e, d_r]
          2) collapse c by summing over c, and permute so ℓ is first:
             W[ℓ,i,j,k] = ∑_{c} W_int[c,i,j,k,ℓ]
             → final W has shape [d_r, d_e, d_e, d_e]
        """
        de = self.d_e
        dr = self.d_r

        if self.ary == 4:
            # Step 1: merge level-1 and level-2
            W_int = torch.einsum(
                'cab, aij, bkl -> cijkl',
                self.ht_internal,  # [r3, r1, r2]
                self.ht_left,      # [r1, de, de]
                self.ht_right      # [r2, de, de]
            )
            # W_int is [r3, de, de, de, de]

            # Step 2a: flatten to [r3, d_e⁴]
            r3, _, _, _, _ = W_int.shape
            W_int_flat = W_int.view(r3, de**4)  # [r3, de^4]

            # Step 2b: mm with ht_root [d_r, r3] → [d_r, d_e^4]
            W_flat = torch.mm(self.ht_root, W_int_flat)  # [dr, de^4]

            # Step 2c: reshape to [d_r, de, de, de, de]
            W = W_flat.view(dr, de, de, de, de)  # [dr, de, de, de, de]
            return W

        else:
            # arity=3 case
            # Step 1: merge level-1a and level-1b through level-2
            W_int = torch.einsum(
                'cab, aij, bkl -> cijkl',
                self.ht_internal,  # [r3, r1, r2]
                self.ht_left,      # [r1, de, de]
                self.ht_right      # [r2, de, dr]
            )
            # W_int shape = [r3, de, de, de, dr]

            # Step 2: sum over c and bring ℓ(=relation) to front:
            #   W[c,i,j,k,ℓ] → W_sum[ℓ,i,j,k]
            W = torch.einsum('cijkl->lijk', W_int)  # [dr, de, de, de]
            return W

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        """
        r_idx          : [B] relation indices
        e_idx          : tuple of three 1D tensors, indices of the known entity slots
        miss_ent_domain: int in {1,2,3,4} indicating which entity is missing
        W              : optionally precomputed HT core; if None, build it
        """
        B   = r_idx.size(0)
        de  = self.d_e
        dr  = self.d_r

        # 1) build or reuse the core
        if W is None:
            W = self.build_W()
            # If arity=4, W is [dr, de, de, de, de]
            # If arity=3, W is [dr, de, de, de]

        # 2) slice out each batch’s core based on r_idx
        r_emb = self.bnr(self.R(r_idx))  # [B, d_r]

        if self.ary == 4:
            # flatten W to [dr, de^4], then mm to get [B, de^4]
            W_mat = torch.mm(r_emb, W.view(dr, -1))  # [B, de^4]
            W_mat = W_mat.view(B, de, de, de, de)     # [B, de, de, de, de]

            # 3a) gather & normalize/dropout the known 3 entity embeddings
            if   miss_ent_domain == 1:
                # known: e2, e3, e4
                e2, e3, e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                # contract out dims j,k,l → keep i
                W_out = torch.einsum('bijkl,bj,bk,bl->bi', W_mat, e2, e3, e4)  # [B, de]
            elif miss_ent_domain == 2:
                # known: e1, e3, e4
                e1, e3, e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                W_out = torch.einsum('bijkl,bi,bk,bl->bj', W_mat, e1, e3, e4)  # [B, de]
            elif miss_ent_domain == 3:
                # known: e1, e2, e4
                e1, e2, e4 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                W_out = torch.einsum('bijkl,bi,bj,bl->bk', W_mat, e1, e2, e4)  # [B, de]
            else:  # miss_ent_domain == 4
                # known: e1, e2, e3
                e1, e2, e3 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                W_out = torch.einsum('bijkl,bi,bj,bk->bl', W_mat, e1, e2, e3)  # [B, de]

        else:  # self.ary == 3
            # W is [dr, de, de, de], flatten to [dr, de^3]
            W_mat = torch.mm(r_emb, W.view(dr, -1))  # [B, de^3]
            W_mat = W_mat.view(B, de, de, de)        # [B, de, de, de]

            # 3b) gather & normalize/dropout known entity embeddings:
            if   miss_ent_domain == 1:
                # known: e2, e3
                e2, e3 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                # contract k→e3, j→e2 → leave i
                W_out = torch.einsum('bijk,ik,ij->bi', W_mat, e3, e2)  # [B, de]
            elif miss_ent_domain == 2:
                # known: e1, e3
                e1, e3 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                W_out = torch.einsum('bijk,ij,ik->bj', W_mat, e1, e3)  # [B, de]
            else:  # miss_ent_domain == 3
                # known: e1, e2
                e1, e2 = [self.input_dropout(self.bne(self.E(idx))) for idx in e_idx]
                W_out = torch.einsum('bijk,ij,ik->bk', W_mat, e1, e2)  # [B, de]

        # 4) final batchnorm, dropout, score
        W_out = self.bnw(W_out)               # [B, de]
        W_out = self.hidden_dropout(W_out)    # [B, de]
        x     = torch.mm(W_out, self.E.weight.t())  # [B, #entities]
        pred  = F.softmax(x, dim=1)           # [B, #entities]

        return pred, W

class GETD_HT(nn.Module):
    def __init__(self, d, d_e, d_r, k, ni, ranks, device, **kwargs):
        super(GETD_HT, self).__init__()
        # entity & relation embeddings
        self.E = nn.Embedding(len(d.entities), d_e, padding_idx=0)
        self.R = nn.Embedding(len(d.relations), d_r, padding_idx=0)
        nn.init.normal_(self.E.weight, 0, 1e-3)
        nn.init.normal_(self.R.weight, 0, 1e-3)

        # dropouts & batch-norms
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.2))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.2))
        self.bne = nn.BatchNorm1d(d_e)
        self.bnr = nn.BatchNorm1d(d_r)
        self.bnw = nn.BatchNorm1d(d_e)

        # we only support arity = 4 here
        assert len(d.train_data[0]) - 1 == 4, "This HT module is arity-4 only"
        self.ary = 4

        # HT rank
        r = ranks

        # Level-1 cores: left merges (e1,e2), right merges (e3,e4)
        #   ht_left[a,i,j]  merges embedding dims i,j → rank-a
        #   ht_right[b,k,l] merges embedding dims k,l → rank-b
        self.ht_left     = nn.Parameter(torch.randn(r, d_e, d_e) * 1e-1)
        self.ht_right    = nn.Parameter(torch.randn(r, d_e, d_e) * 1e-1)

        # Level-2 core: merges those two rank vectors → an internal rank-c
        self.ht_internal = nn.Parameter(torch.randn(r, r, r) * 1e-1)

        # Root core: merges internal rank-c with the relation embedding dim d_r
        self.ht_root     = nn.Parameter(torch.randn(d_r, r) * 1e-1)

        self.loss = MyLoss()

    def build_W(self):
        """
        Build the full 5-D weight tensor W[d,i,j,k,l] via HT:
          1) contract ht_left & ht_right through ht_internal → W_int[c,i,j,k,l]
          2) contract W_int with ht_root → W[d,i,j,k,l]
        """
        # 1) cijkl = sum_{a,b} ht_internal[c,a,b] * ht_left[a,i,j] * ht_right[b,k,l]
        W_int = torch.einsum(
            'cab, aij, bkl -> cijkl',
            self.ht_internal,  # [r,    r,    r]
            self.ht_left,      # [r,    de,   de]
            self.ht_right      # [r,    de,   de]
        )
        # 2) dijkl = sum_c ht_root[d,c] * W_int[c,i,j,k,l]
        W = torch.einsum(
            'dc, cijkl -> dijkl',
            self.ht_root,   # [dr,   r]
            W_int           # [r,    de,   de,   de,   de]
        )
        # result: [dr, de, de, de, de]
        return W

    def forward(self, r_idx, e_idx, miss_ent_domain, W=None):
        B = r_idx.size(0)
        de = self.E.embedding_dim
        dr = self.R.embedding_dim

        # 1) build (or reuse) the full core
        if W is None:
            W = self.build_W()               # [dr, de, de, de, de]

        # 2) condition on relation: each batch gets its own slice of W
        r_emb = self.bnr(self.R(r_idx))       # [B, dr]
        W_mat = torch.mm(r_emb, W.view(dr, -1))  # [B, de^4]
        W_mat = W_mat.view(B, de, de, de, de)     # [B, de, de, de, de]

        # 3) pull out the known entity embeddings & normalize/dropout
        #    e_idx is always a triplet of the *known* slots in the order e_idx[0],e_idx[1],e_idx[2]
        e1 = e2 = e3 = e4 = None
        if   miss_ent_domain == 1:
            # known: e2,e3,e4
            e2,e3,e4 = [ self.input_dropout(self.bne(self.E(idx))) for idx in e_idx ]
        elif miss_ent_domain == 2:
            # known: e1,e3,e4
            e1,e3,e4 = [ self.input_dropout(self.bne(self.E(idx))) for idx in e_idx ]
        elif miss_ent_domain == 3:
            # known: e1,e2,e4
            e1,e2,e4 = [ self.input_dropout(self.bne(self.E(idx))) for idx in e_idx ]
        elif miss_ent_domain == 4:
            # known: e1,e2,e3
            e1,e2,e3 = [ self.input_dropout(self.bne(self.E(idx))) for idx in e_idx ]
        else:
            raise ValueError("miss_ent_domain must be 1..4")

        # 4) contract out the known three entity dims, leaving a [B, de] score vector
        if   miss_ent_domain == 1:
            # W_mat[b,i,j,k,l], contract j,k,l with e2,e3,e4 → leave i
            W_out = torch.einsum('bijkl,bj,bk,bl->bi',
                                W_mat, e2, e3, e4)
        elif miss_ent_domain == 2:
            # leave j
            W_out = torch.einsum('bijkl,bi,bk,bl->bj',
                                W_mat, e1, e3, e4)
        elif miss_ent_domain == 3:
            # leave k
            W_out = torch.einsum('bijkl,bi,bj,bl->bk',
                                W_mat, e1, e2, e4)
        else:  # miss_ent_domain == 4
            # leave l
            W_out = torch.einsum('bijkl,bi,bj,bk->bl',
                                W_mat, e1, e2, e3)

        # 5) batch-norm, dropout, then score against all entity embeddings
        W_out = self.bnw(W_out)                 # [B, de]
        W_out = self.hidden_dropout(W_out)      # [B, de]
        x = torch.mm(W_out, self.E.weight.t())  # [B, #entities]
        pred = F.softmax(x, dim=1)

        return pred, W



class GETD_TT(torch.nn.Module):
    def __init__(self, d, d_e, d_r, k, ni_list, ranks_list, device, **kwargs):
        super(GETD_TT, self).__init__()

        assert len(ni_list) == k, "ni_list length should be equal to k"
        assert len(ranks_list) == k+1, \
            "In a TT of order k, you must supply k+1 ranks (including the two boundary ranks=1)."
        assert ranks_list[0] == 1 and ranks_list[-1] == 1, \
            "In TT, the first and last rank must be 1. Got {}".format(ranks_list)

        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        
        # Customizable ni_list per dimension
        self.Zlist = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1e-1, 1e-1, (ranks_list[i], ni_list[i], ranks_list[i+1])),
                             dtype=torch.float, requires_grad=True).to(device)
            ) for i in range(k)
        ])

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
                einsum_str = 'aib,bjc,ckd,dlf->ijkl'
            elif k == 5:
                einsum_str = 'aib,bjc,ckd,dle,emf->ijklm'
            else:
                raise ValueError("TR equation for k={} is not defined.".format(k))
            
            W0 = torch.einsum(einsum_str, *Zlist)
            
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



class GETD_FC(nn.Module):
    """
    Fully-connected Tensor-Ring GETD model with on-the-fly contraction.

    Modes: 0 = relation, 1..k-1 = entities (k total modes).
    Cores: one per mode, each tensor shaped by bond dimensions and a physical dimension.
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

        self.E = torch.nn.Embedding(len(data.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(data.relations), embedding_dim=d_r, padding_idx=0)
        
        self.E.weight.data = (1e-3 * torch.randn((len(data.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3 * torch.randn((len(data.relations), d_r), dtype=torch.float).to(device))
        # Core tensors
        # self.cores = nn.ParameterList()
        # for i in range(self.k):
        #     shape = []
        #     for j in range(self.k):
        #         if i == j: continue
        #         edge = (i, j) if i < j else (j, i)
        #         shape.append(self.bond_ranks[edge])
        #     shape.append(self.ni_list[i])
        #     G = nn.Parameter(torch.randn(*shape, device=device) * 1e-2)
            
        #     self.cores.append(G)
            
            
        # Core tensors
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
        # Norms & dropout
        
        self.bnr = nn.BatchNorm1d(d_r)
        self.bne = nn.BatchNorm1d(d_e)
        self.bnw = nn.BatchNorm1d(d_e)
        self.input_dropout  = nn.Dropout(kwargs.get("input_dropout", 0.0))
        self.hidden_dropout = nn.Dropout(kwargs.get("hidden_dropout", 0.0))
        self.loss = MyLoss()
        
    def forward(self, r_idx, e_idx_list, miss, W=None):
        B = r_idx.size(0)
        #print(f"\n=== RUN miss={miss}  batch={B} ===")

        # subscripts for einsum
        bond_letters = {e: chr(ord('a')+i) for i,e in enumerate(self.edges)}
        phys_letters = [chr(ord('A')+m) for m in range(self.k)]

        # 1) embed relation + fuse G0
        R0 = self.R(r_idx)
        #print(f"[1] R0.shape = {tuple(R0.shape)}")
        R0 = self.bnr(R0)
        R0 = self.input_dropout(R0)
        G0 = self.cores[0]
        p0 = phys_letters[0]
        b0 = [bond_letters[e] for e in self.edges if 0 in e]
        eq0 = f"z{p0},{''.join(b0)+p0}->z{''.join(b0)}"
        #print(f"[2] Fuse G0: G0.shape={tuple(G0.shape)}  einsum='{eq0}'")
        T = torch.einsum(eq0, R0, G0)
        sub = 'z' + ''.join(b0)
        #print(f"[2] → T.shape={tuple(T.shape)}, sub='{sub}'")

        # 2) fuse all non-missing entity cores (keep all bonds)
        for m in range(1, self.k):
            if m == miss: continue
            Gm = self.cores[m]
            pm = phys_letters[m]
            bm = [bond_letters[e] for e in self.edges if m in e]
            eqf = f"{sub},{''.join(bm)+pm}->{sub+pm}"
            #print(f"[3.{m}] Fuse G{m}: shape={tuple(Gm.shape)}  einsum='{eqf}'")
            T = torch.einsum(eqf, T, Gm)
            sub += pm
            #print(f"[3.{m}] → T.shape={tuple(T.shape)}, sub='{sub}'")

        # 3) absorb all non-missing embeddings (collapse phys legs only)
        ei = 0
        for m in range(1, self.k):
            if m == miss: continue
            E_m = self.E(e_idx_list[ei]); ei += 1
            pm = phys_letters[m]
            eqa = f"{sub},z{pm}->{sub.replace(pm, '')}"
            #print(f"[4.{m}] Absorb E{m}: E.shape={tuple(E_m.shape)}  einsum='{eqa}'")
            E_m = self.bne(E_m)
            E_m = self.input_dropout(E_m)
            T = torch.einsum(eqa, T, E_m)
            sub = sub.replace(pm, '')
            #print(f"[4.{m}] → T.shape={tuple(T.shape)}, sub='{sub}'")

        # 4) fuse missing core + collapse
        Gm = self.cores[miss]
        pm = phys_letters[miss]
        bm = [bond_letters[e] for e in self.edges if miss in e]
        eqm = f"{sub},{''.join(bm)+pm}->{sub+pm}"
        #print(f"[5] Fuse missing G{miss}: shape={tuple(Gm.shape)}  einsum='{eqm}'")
        T = torch.einsum(eqm, T, Gm)
        sub += pm
        #print(f"[5] → T.shape={tuple(T.shape)}, sub='{sub}'")

        # collapse all bond dims
        S = T
        for _ in range(S.dim()-2):
            S = S.sum(dim=1)
        #print(f"[6] After collapse S.shape={tuple(S.shape)}")

        # final BN+dropout+softmax
        out = self.bnw(S)
        out = self.hidden_dropout(out)
        logits = out @ self.E.weight.t()
        return F.softmax(logits, dim=1), W

    # def forward(self, r_idx, e_idx_list, miss_ent_domain, W=None):
    #     """
    #     r_idx:           (B,)   relation indices
    #     e_idx_list:      list of (k-2) tensors each (B,) for the known entities
    #     miss_ent_domain: int ∈ [1..k-1]
    #     """
    #     B = r_idx.size(0)
    #     #print(f"\n[DEBUG] B={B}, missing mode={miss_ent_domain}")
    #     # 1) embed relation
    #     r = self.R(r_idx)         # (B, d_r)
    #     #print(f"[1] r.shape = {tuple(r.shape)}")
    #     r = self.bnr(r)
    #     r = self.input_dropout(r)

    #     # letter assignment for einsum
    #     bond_letters = {e: chr(ord('a')+i) for i,e in enumerate(self.edges)}
    #     phys_letters = [chr(ord('i')+m) for m in range(self.k)]

    #     # 2) fuse relation into core G₀
    #     G0 = self.cores[0]
    #     p0    = phys_letters[0]
    #     bonds0= [bond_letters[e] for e in self.edges if 0 in e]
    #     eq0   = f"z{p0},{''.join(bonds0)+p0}->z{''.join(bonds0)}"
    #     #print(f"[2] Fuse G0: G0.shape={tuple(G0.shape)}, einsum='{eq0}'")
    #     T     = torch.einsum(eq0, r, G0)
    #     sub_T = 'z' + ''.join(bonds0)
    #     #print(f"[2] T.shape={tuple(T.shape)}, sub_T='{sub_T}'")
    #     # 3) fuse & absorb all non‑missing entity cores
    #     idx_known = 0
    #     for mode in range(1, self.k):
    #         if mode == miss_ent_domain:
    #             continue

    #         # fuse core G_mode
    #         Gm = self.cores[mode]
    #         pm = phys_letters[mode]
    #         bonds_m = [bond_letters[e] for e in self.edges if mode in e]
    #         eq_f = f"{sub_T},{''.join(bonds_m)+pm}->{sub_T+pm}"
    #         #print(f"[3.{mode}] Fuse G{mode}: G.shape={tuple(Gm.shape)}, einsum='{eq_f}'")
    #         T   = torch.einsum(eq_f, T, Gm)
    #         #print(f"[3.{mode}] After fuse, T.shape={tuple(T.shape)}")
    #         # absorb known entity embedding
    #         e = self.E(e_idx_list[idx_known]);  idx_known += 1
    #         e = self.bne(e)
    #         e = self.input_dropout(e)
    #         eq_a = f"{sub_T+pm},z{pm}->{sub_T}"
    #         #print(f"[3.{mode}] Absorb e{mode}: e.shape={tuple(e.shape)}, einsum='{eq_a}'")
    #         T   = torch.einsum(eq_a, T, e)
    #         #print(f"[3.{mode}] After absorb, T.shape={tuple(T.shape)}")
    #     # 4) now project into the missing mode
    #     # fuse missing core
    #     Gm = self.cores[miss_ent_domain]
    #     pm = phys_letters[miss_ent_domain]
    #     bonds_m = [bond_letters[e] for e in self.edges if miss_ent_domain in e]
    #     eq_fm = f"{sub_T},{''.join(bonds_m)+pm}->{sub_T+pm}"
    #     #print(f"[4] Fuse missing G{miss_ent_domain}: G.shape={tuple(Gm.shape)}, einsum='{eq_fm}'")
    #     T    = torch.einsum(eq_fm, T, Gm)
    #     #print(f"[4] After missing fuse, T.shape={tuple(T.shape)}")
    #     # collapse all bond dims → S[b, i_miss]
    #     S = T
    #     for _ in range(S.dim()-2):
    #         S = S.sum(dim=1)
    #     #print(f"[4] After collapse, S.shape={tuple(S.shape)}")
    #     # final prediction
    #     out    = self.bnw(S)
    #     out    = self.hidden_dropout(out)
    #     logits = out @ self.E.weight.t()     # (B, n_entities)
    #     return F.softmax(logits, dim=1), W

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