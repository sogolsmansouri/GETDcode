# TNmodel.py
# -----------------
import os
import random
import string
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(graph, filename="best_graph.png"):
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=1500, font_size=14)
    plt.title("Best Tensor Network Graph")
    plt.savefig(filename)
    plt.close()


class MyLoss(nn.Module):
    def forward(self, pred, target):
        return nn.functional.binary_cross_entropy_with_logits(pred, target)


class TensorNetworkDecoder(nn.Module):
    def __init__(self, G, dims, ranks, device):
        super().__init__()
        self.G = G
        self.device = device
        letters = string.ascii_lowercase
        self.node_letter = {v: letters[i] for i, v in enumerate(sorted(G.nodes()))}
        aux = letters[len(G.nodes()):]
        self.edge_letter = {tuple(sorted(e)): aux[i] for i, e in enumerate(sorted(G.edges()))}
        self.neighbors = {v: sorted(G.neighbors(v)) for v in sorted(G.nodes())}
        self.cores = nn.ParameterDict()
        for v in sorted(G.nodes()):
            shape = [dims[v]] + [ranks[tuple(sorted((v, n)))] for n in self.neighbors[v]]
            self.cores[str(v)] = nn.Parameter(torch.randn(*shape, device=device) * 1e-3)
        self.subs = []
        for v in sorted(G.nodes()):
            sub = self.node_letter[v] + "".join(
                self.edge_letter[tuple(sorted((v, n)))] for n in self.neighbors[v]
            )
            self.subs.append(sub)

    def forward(self, r_embed, entity_embeds, miss):
        core_ops = [self.cores[str(v)] for v in sorted(self.G.nodes())]
        rem = list(entity_embeds)
        embed_map = {}
        for v in sorted(self.G.nodes()):
            if v == 0 or v == miss:
                continue
            if rem:
                embed_map[v] = rem.pop(0)
        emb_ops, embed_letters = [], []
        for v in sorted(self.G.nodes()):
            if v == 0:
                emb_ops.append(r_embed)
                embed_letters.append(f"...{self.node_letter[v]}")
            elif v in embed_map:
                emb_ops.append(embed_map[v])
                embed_letters.append(f"...{self.node_letter[v]}")
        eq = ",".join(self.subs + embed_letters) + f"->...{self.node_letter[miss]}"
        return torch.einsum(eq, *core_ops, *emb_ops)


class TNModel(nn.Module):
    def __init__(
        self,
        num_entities,
        num_relations,
        d_e,
        d_r,
        decoder_config,
        device,
        hidden_dropout=0.3,
        ent_emb_path=None,
        rel_emb_path=None
    ):
        super().__init__()
        # Embeddings
        self.E = nn.Embedding(num_entities, d_e, padding_idx=0)
        self.R = nn.Embedding(num_relations, d_r, padding_idx=0)
        # Load pretrained if provided
        if ent_emb_path and os.path.exists(ent_emb_path):
            ent_np = np.load(ent_emb_path)
            self.E.weight.data.copy_(torch.from_numpy(ent_np).to(device))
        else:
            nn.init.uniform_(self.E.weight, -0.1, 0.1)
        if rel_emb_path and os.path.exists(rel_emb_path):
            rel_np = np.load(rel_emb_path)
            self.R.weight.data.copy_(torch.from_numpy(rel_np).to(device))
        else:
            nn.init.uniform_(self.R.weight, -0.1, 0.1)
        # Preserve pretrained geometry
        self.embed_norm = nn.LayerNorm(d_e)
        self.rel_norm = nn.LayerNorm(d_r)
        # Decoder
        G = decoder_config['graph']
        dims = {0: d_r, **{i: d_e for i in range(1, decoder_config['arity'] + 1)}}
        self.decoder = TensorNetworkDecoder(G, dims, decoder_config['ranks'], device)
        # Dropout after scoring
        self.hid_dp = nn.Dropout(hidden_dropout)
        self.arity = decoder_config['arity']

    def forward(self, r_idx, e_idx, miss):
        r = self.rel_norm(self.R(r_idx))
        es = [self.embed_norm(self.E(idx)) for i, idx in enumerate(e_idx, start=1) if i != miss]
        raw = self.decoder(r, es, miss)
        logits = raw @ self.E.weight.t()
        return self.hid_dp(logits)


# GeneticSearch unchanged
class GeneticSearch:
    def __init__(self, pop_size, arity, fitness_fn, generations=3):
        self.pop_size = pop_size
        self.arity = arity
        self.fitness_fn = fitness_fn
        self.generations = generations

    def _random_graph(self):
        G = nx.Graph()
        nodes = list(range(self.arity + 1))
        G.add_nodes_from(nodes)
        for i in range(1, self.arity + 1):
            G.add_edge(i, random.randint(0, i - 1))
        if random.random() < 0.3:
            G.add_edge(*random.sample(nodes, 2))
        return G

    def _random_chromosome(self):
        G = self._random_graph()
        return {"graph": G,
                "arity": self.arity,
                "ranks": {tuple(sorted(e)): random.randint(10, 50) for e in G.edges()}}

    def evolve(self):
        print("🧬 Starting Genetic Search…")
        pop = [self._random_chromosome() for _ in range(self.pop_size)]
        all_results = []
        for gen in range(1, self.generations + 1):
            print(f"— Generation {gen} —")
            scored = [(c, self.fitness_fn(c)) for c in pop]
            scored.sort(key=lambda x: x[1], reverse=True)
            torch.cuda.empty_cache()
            all_results.extend(scored)
            survivors = [c for c, _ in scored[:max(2, len(pop)//2)]]
            children = []
            while len(children) < self.pop_size - len(survivors):
                p1, p2 = random.sample(survivors, 2)
                child = random.choice([p1, p2]).copy()
                if random.random() < 0.3:
                    child['graph'] = self._random_graph()
                    child['ranks'] = {tuple(sorted(e)): random.randint(10, 50) for e in child['graph'].edges()}
                else:
                    e = random.choice(list(child['ranks'].keys()))
                    child['ranks'][e] = random.randint(10, 50)
                children.append(child)
            pop = survivors + children
        all_results.sort(key=lambda x: x[1], reverse=True)
        best, _ = all_results[0]
        visualize_graph(best['graph'])
        global best_ranks; best_ranks = best['ranks']
        return best

