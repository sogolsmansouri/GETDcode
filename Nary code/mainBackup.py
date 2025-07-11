import os
import json
import argparse
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from load_data import Data
from TNmodel import TNModel, GeneticSearch
from model import GETD, GETD_TT, GETD_TN
from collections import defaultdict

def fitness_fn_builder(exp, d, args, device):
    def fitness_fn(cfg):
        model = TNModel(
            num_entities=len(d.entities),
            num_relations=len(d.relations),
            d_e=args.edim,
            d_r=args.rdim,
            decoder_config=cfg,
            device=device,
            hidden_dropout=args.hidden_dropout,
            ent_emb_path=args.ent_emb_path,
            rel_emb_path=args.rel_emb_path
        ).to(device)
        optimizer = Adam([
            {'params': model.E.parameters(), 'lr': args.emb_lr},
            {'params': [p for n,p in model.named_parameters() if 'E.weight' not in n], 'lr': args.lr}
        ])
        model.train()
        idxs = exp.get_data_idxs(d.train_data)
        # one-batch-per-link evaluation
        for miss in range(1, cfg['arity']+1):
            ev = exp.get_er_vocab(idxs, miss)
            pairs = list(ev.keys())[:args.batch_size]
            if not pairs: continue
            arr = np.array(pairs)
            ridx = torch.tensor(arr[:,0], device=device)
            e_idx = [torch.tensor(arr[:,j], device=device) for j in range(1,arr.shape[1])]
            pred = model(ridx, e_idx, miss)
            tgt = exp.get_batch(ev, pairs, 0)[1]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            return exp.evaluate(model, d.valid_data)[0]
    return fitness_fn


class Experiment:
    def __init__(
        self,
        num_iterations, batch_size, learning_rate, decay_rate,
        ent_vec_dim, rel_vec_dim, k, ni, ranks,
        input_dropout, hidden_dropout, method
    ):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.k = k
        self.ni = ni
        self.ranks = ranks
        self.kwargs = {'input_dropout': input_dropout, 'hidden_dropout': hidden_dropout}
        self.method = method.upper()

    def get_data_idxs(self, data):
        ary = len(data[0]) - 1
        idxs = []
        for row in data:
            base = [self.relation_idxs[row[0]]]
            entities = [self.entity_idxs[e] for e in row[1:]]
            idxs.append(tuple(base + entities))
        return idxs

    def get_er_vocab(self, data_idxs, miss):
        ev = defaultdict(list)
        for t in data_idxs:
            key = list(t)
            val = key.pop(miss)
            ev[tuple(key)].append(val)
        return ev

    def get_batch(self, ev, pairs, idx):
        batch = pairs[idx: idx + self.batch_size]
        B = len(batch)
        targets = torch.zeros(B, len(self.entity_idxs), device=self.device)
        for i, p in enumerate(batch):
            for v in ev[p]:
                targets[i, v] = 1.0
        return batch, targets

    def evaluate(self, model, data, W=None):
        model.eval()
        test = self.get_data_idxs(data)
        ary = len(test[0]) - 1
        hits = {1: [], 3: [], 10: []}
        ranks = []
        with torch.no_grad():
            for t in test:
                ridx = torch.tensor([t[0]], device=self.device)
                for miss in range(1, ary + 1):
                    e_list = list(t[1:])
                    true = e_list.pop(miss - 1)
                    e_idx = [torch.tensor([e], device=self.device) for e in e_list]
                    pred = model(ridx, e_idx, miss)
                    sorted_idx = torch.argsort(pred, dim=1, descending=True)[0]
                    rank = (sorted_idx == true).nonzero().item() + 1
                    ranks.append(rank)
                    for k in hits:
                        hits[k].append(1.0 if rank <= k else 0.0)
        model.train()
        mrr = np.mean(1.0 / np.array(ranks))
        return mrr, np.mean(hits[10]), np.mean(hits[3]), np.mean(hits[1])

    # def evaluate_test(self, model, data):
    #     print("Evaluating on test set...")
    #     mrr, h10, h3, h1 = self.evaluate(model, data)
    #     print(f"Test MRR={mrr:.4f} Hits@10={h10:.4f} Hits@3={h3:.4f} Hits@1={h1:.4f}")
    #     return mrr, h10, h3, h1
    def evaluate_test(self, model, d):
        """
        Evaluate the given model on the test set of dataset `d`.
        """
        print("Evaluating on test set...")
        mrr, h10, h3, h1 = self.evaluate(model, d.test_data)
        print(f"Test MRR={mrr:.4f} Hits@10={h10:.4f} Hits@3={h3:.4f} Hits@1={h1:.4f}")
        return mrr, h10, h3, h1


    def train_and_eval(self, model, d):
        self.device = next(model.parameters()).device
        self.entity_idxs = {e: i for i, e in enumerate(d.entities)}
        self.relation_idxs = {r: i for i, r in enumerate(d.relations)}
        train_idxs = self.get_data_idxs(d.train_data)
        ary = len(train_idxs[0]) - 1
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)
        best_iter, best_mrr = 0, -1
        for epoch in range(1, self.num_iterations + 1):
            losses = []
            np.random.shuffle(train_idxs)
            for miss in range(1, ary + 1):
                ev = self.get_er_vocab(train_idxs, miss)
                pairs = list(ev.keys())
                for i in range(0, len(pairs), self.batch_size):
                    batch, tgt = self.get_batch(ev, pairs, i)
                    arr = np.array(batch)
                    ridx = torch.tensor(arr[:, 0], device=self.device)
                    e_idx = [torch.tensor(arr[:, j], device=self.device) for j in range(1, arr.shape[1])]
                    pred = model(ridx, e_idx, miss)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, tgt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    losses.append(loss.item())
            if self.decay_rate: scheduler.step()
            mrr, h10, h3, h1 = self.evaluate(model, d.valid_data)
            print(f"Epoch {epoch} train loss={np.mean(losses):.4f} | valid MRR={mrr:.4f}")
            if mrr > best_mrr:
                best_mrr, best_iter = mrr, epoch
                self.best_model = model
            if epoch - best_iter >= 10:
                print(f"Early stopping at epoch {epoch}, best MRR={best_mrr:.4f} at epoch {best_iter}")
                break

    def save_embeddings(self, model, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        ent = model.E.weight.detach().cpu().numpy()
        rel = model.R.weight.detach().cpu().numpy()
        np.save(os.path.join(out_dir, "entities.npy"), ent)
        np.save(os.path.join(out_dir, "relations.npy"), rel)
        print(f"Saved embeddings to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="JF17K-3")
    parser.add_argument("--method", choices=["TR","TN"], required=True)
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--emb_lr", type=float, default=0.0001)
    parser.add_argument("--dr", type=float, default=0.995)
    parser.add_argument("--edim", type=int, default=25)
    parser.add_argument("--rdim", type=int, default=25)
    parser.add_argument("--input_dropout", type=float, default=0.3)
    parser.add_argument("--hidden_dropout", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_i", type=int, default=25)
    parser.add_argument("--TR_ranks", type=int, default=30)
    parser.add_argument("--ent_emb_path", type=str, default="embeddings/entities.npy")
    parser.add_argument("--rel_emb_path", type=str, default="embeddings/relations.npy")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="embeddings")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); np.random.seed(42)

    d = Data(f"./data/{args.dataset}/")
    exp = Experiment(
        args.num_iterations, args.batch_size,
        args.lr, args.dr,
        args.edim, args.rdim,
        args.k, args.n_i, args.TR_ranks,
        args.input_dropout, args.hidden_dropout, args.method
    )
    exp.device = device
    exp.entity_idxs   = {e:i for i,e in enumerate(d.entities)}
    exp.relation_idxs = {r:i for i,r in enumerate(d.relations)}

    if args.method == "TN":
        if args.search:
            best_cfg = GeneticSearch(
                pop_size=5,
                arity=len(d.train_data[0]) - 1,
                fitness_fn=fitness_fn_builder(exp, d, args, device),
                generations=3
            ).evolve()
        else:
            best_cfg = json.load(open("best_config.json"))
        model = TNModel(
            num_entities=len(d.entities),
            num_relations=len(d.relations),
            d_e=args.edim,
            d_r=args.rdim,
            decoder_config=best_cfg,
            device=device,
            hidden_dropout=args.hidden_dropout,
            ent_emb_path=args.ent_emb_path,
            rel_emb_path=args.rel_emb_path
        ).to(device)
    else:
        # TR baseline
        model = GETD(
            d, args.edim, args.rdim,
            k=args.k, ni=args.n_i, ranks=args.TR_ranks,
            device=device, input_dropout=0.0, hidden_dropout=0.0
        ).to(device)

    # Two-tier optimizer and scheduler
    optimizer = Adam([
        {'params': model.E.parameters(), 'lr': args.emb_lr},
        {'params': [p for n,p in model.named_parameters() if 'E.weight' not in n], 'lr': args.lr}
    ])
    scheduler = ExponentialLR(optimizer, gamma=args.dr)

    # Warm-freeze embeddings
    for p in model.E.parameters(): p.requires_grad = False

    for epoch in range(1, args.num_iterations+1):
        if epoch == args.freeze_epochs+1:
            for p in model.E.parameters(): p.requires_grad = True
        exp.train_and_eval(model, d)
        scheduler.step()

    exp.evaluate_test(model, d)
    exp.save_embeddings(model, args.out_dir)

if __name__ == "__main__":
    main()
