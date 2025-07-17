from load_data import Data
import numpy as np
import torch
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import os

device = torch.device('cuda:0')

class Experiment:
    def __init__(self, num_iterations, batch_size, learning_rate, decay_rate, ent_vec_dim, rel_vec_dim,
                 k, ni, ranks, input_dropout, hidden_dropout):
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

    def get_data_idxs(self, data):
        if len(data[0])-1 == 3:
            data_idxs = [(self.relation_idxs[data[i][0]], self.entity_idxs[data[i][1]], self.entity_idxs[data[i][2]], self.entity_idxs[data[i][3]]) for i in range(len(data))]
        elif len(data[0])-1 == 4:
            data_idxs = [(self.relation_idxs[data[i][0]], self.entity_idxs[data[i][1]], self.entity_idxs[data[i][2]], self.entity_idxs[data[i][3]], self.entity_idxs[data[i][4]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data, miss_ent_domain):
        er_vocab = defaultdict(list)
        if len(data[0])-1 == 3:
            if miss_ent_domain == 1:
                for triple in data:
                    er_vocab[(triple[0], triple[2], triple[3])].append(triple[1])
            elif miss_ent_domain == 2:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[3])].append(triple[2])
            elif miss_ent_domain == 3:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2])].append(triple[3])
        elif len(data[0])-1 == 4:
            if miss_ent_domain == 1:
                for triple in data:
                    er_vocab[(triple[0], triple[2], triple[3], triple[4])].append(triple[1])
            elif miss_ent_domain == 2:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[3], triple[4])].append(triple[2])
            elif miss_ent_domain == 3:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2], triple[4])].append(triple[3])
            elif miss_ent_domain == 4:
                for triple in data:
                    er_vocab[(triple[0], triple[1], triple[2], triple[3])].append(triple[4])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets).to(device)
        return np.array(batch), targets

    


    def evaluate(self, model, data, W):
        hits, ranks, losses = [], [], []
        for _ in [1, 3, 10]:
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        ary = len(test_data_idxs[0])-1

        er_vocab_list = []
        er_vocab_pairs_list = []
        for miss_ent_domain in range(1, ary+1):
            er_vocab = self.get_er_vocab(self.get_data_idxs(d.data), miss_ent_domain)
            er_vocab_pairs = list(er_vocab.keys())
            er_vocab_list.append(er_vocab)
            er_vocab_pairs_list.append(er_vocab_pairs)

        for miss_ent_domain in range(1, ary+1):
            er_vocab = er_vocab_list[miss_ent_domain-1]
            for i in range(0, len(test_data_idxs), self.batch_size):
                data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
               
                r_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
                e1_idx = torch.tensor(data_batch[:, 1], dtype=torch.long).to(device)
                e2_idx = torch.tensor(data_batch[:, 2], dtype=torch.long).to(device)
                e3_idx = torch.tensor(data_batch[:, 3], dtype=torch.long).to(device)
                if ary == 3:
                    if miss_ent_domain == 1:
                        e_idx = [e2_idx, e3_idx]
                    elif miss_ent_domain == 2:
                        e_idx = [e1_idx, e3_idx]
                    elif miss_ent_domain == 3:
                        e_idx = [e1_idx, e2_idx]
                elif ary == 4:
                    e4_idx = torch.tensor(data_batch[:, 4], dtype=torch.long).to(device)
                    if miss_ent_domain == 1:
                        e_idx = [e2_idx, e3_idx, e4_idx]
                    elif miss_ent_domain == 2:
                        e_idx = [e1_idx, e3_idx, e4_idx]
                    elif miss_ent_domain == 3:
                        e_idx = [e1_idx, e2_idx, e4_idx]
                    elif miss_ent_domain == 4:
                        e_idx = [e1_idx, e2_idx, e3_idx]
                pred, _ = model.forward(r_idx, e_idx, miss_ent_domain, W)

                e_all_idx = []
                for k0 in range(1, ary+1):
                    e_all_idx.append(torch.tensor(data_batch[:, k0], dtype=torch.long).to(device))

                for j in range(data_batch.shape[0]):
                    er_vocab_key = []
                    for k0 in range(ary+1):
                        er_vocab_key.append(data_batch[j][k0])
                    er_vocab_key.remove(data_batch[j][miss_ent_domain])

                    filt = er_vocab[tuple(er_vocab_key)]
                    target_value = pred[j, e_all_idx[miss_ent_domain-1][j]].item()
                    pred[j, filt] = 0.0
                    pred[j, e_all_idx[miss_ent_domain-1][j]] = target_value

                sort_values, sort_idxs = torch.sort(pred, dim=1, descending=True)
                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(data_batch.shape[0]):
                    rank = np.where(sort_idxs[j] == e_all_idx[miss_ent_domain-1][j].item())[0][0]
                    ranks.append(rank+1)
                    for id, hits_level in enumerate([1, 3, 10]):
                        if rank+1 <= hits_level:
                            hits[id].append(1.0)
                        else:
                            hits[id].append(0.0)
        return np.mean(1./np.array(ranks)), np.mean(hits[2]), np.mean(hits[1]), np.mean(hits[0])

    def train_and_eval(self):
        
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        ary = len(d.train_data[0])-1
        
        if args.method == 'TR':
            #model = GETD(d, args.edim, args.rdim, args.k, args.n_i, args.TR_ranks, device, input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout)
            model = GETD(d, args.edim, args.rdim, args.k, args.ni_list, args.rank_list, device,
             input_dropout=args.input_dropout,
             hidden_dropout=args.hidden_dropout)
        elif args.method == 'FC':
            #model = GETD(d, args.edim, args.rdim, args.k, args.n_i, args.TR_ranks, device, input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout)
            model = GETD_FC(d, args.edim, args.rdim, args.ni_list, args.rank_list, device,
             input_dropout=args.input_dropout,
             hidden_dropout=args.hidden_dropout)
            print("Edges:", model.edges)
            for i, G in enumerate(model.cores):
                print(f" Core {i}: shape = {tuple(G.shape)}")
        # elif args.method == 'HT':
        #     model = GETD_HT3(d, args.edim, args.rdim, args.k, args.ni_list, args.TR_ranks, device, input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout)
        
        elif args.method == 'TT':
            model = GETD_TT(d, args.edim, args.rdim, args.k, args.ni_list, args.rank_list, device, input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout)
        elif args.method == 'HT':
            model = GETD_HT3_Enhanced(
                d, 
                args.edim, 
                args.rdim, 
                args.k, 
                args.ni_list, 
                args.ht_ranks, 
                device,
                ht_tree=args.ht_tree,
                input_dropout=args.input_dropout,
                hidden_dropout=args.hidden_dropout,
                mlp_hidden=args.mlp_hidden if hasattr(args,'mlp_hidden') else None
            )
        
        ##model = GETD(d, self.ent_vec_dim, self.rel_vec_dim, self.k, self.ni, self.ranks, device, **self.kwargs)
        import torch
        torch.cuda.empty_cache()

        model = model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        torch.cuda.empty_cache()
        best_valid_iter = 0
        best_valid_metric = {'mrr': -1, 'test_mrr': -1, 'test_hit1': -1, 'test_hit3': -1, 'test_hit10': -1}

        er_vocab_list, er_vocab_pairs_list = [], []
        for miss_ent_domain in range(1, ary+1):
            er_vocab = self.get_er_vocab(train_data_idxs, miss_ent_domain)
            er_vocab_pairs = list(er_vocab.keys())
            er_vocab_list.append(er_vocab)
            er_vocab_pairs_list.append(er_vocab_pairs)

        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            for er_vocab_pairs in er_vocab_pairs_list:
                np.random.shuffle(er_vocab_pairs)

            for miss_ent_domain in range(1, ary+1):
                er_vocab = er_vocab_list[miss_ent_domain-1]
                er_vocab_pairs = er_vocab_pairs_list[miss_ent_domain-1]

                for j in range(0, len(er_vocab_pairs), self.batch_size):
                    data_batch, label = self.get_batch(er_vocab, er_vocab_pairs, j)
                    
                    # Save last label row for debug print in forward
                    model.last_label_row = label[0, :10].cpu().numpy()
                    opt.zero_grad()
                    # r_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
                    # e1_idx = torch.tensor(data_batch[:, 1], dtype=torch.long).to(device)
                    # e2_idx = torch.tensor(data_batch[:, 2], dtype=torch.long).to(device)
                    # print("Sample data_batch shape:", data_batch.shape)
                    # print("Sample data_batch row:", data_batch[0])
                    # if ary == 3:
                    #     e_idx = [e1_idx, e2_idx]
                    # elif ary == 4:
                    #     e3_idx = torch.tensor(data_batch[:, 3], dtype=torch.long).to(device)
                    #     e_idx = [e1_idx, e2_idx, e3_idx]
                    r_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
                    e_idx = [torch.tensor(data_batch[:, i], dtype=torch.long).to(device) for i in range(1, data_batch.shape[1])]
                    # print("Sample data_batch shape:", data_batch.shape)
                    # print("Sample data_batch row:", data_batch[0])

                    
                    
                    # r_idx = torch.tensor(data_batch[:, 0], dtype=torch.long).to(device)
                    # num_columns = data_batch.shape[1]
                    # e_idx = [torch.tensor(data_batch[:, i], dtype=torch.long).to(device) for i in range(1, num_columns)]
                    


                    pred, W = model.forward(r_idx, e_idx, miss_ent_domain)
                    pred = pred.to(device)
                    loss = model.loss(pred, label)
                    loss.backward()
                    opt.step()
                    torch.cuda.empty_cache()
                    losses.append(loss.item())

            print('\nEpoch %d train, loss=%f' % (it, np.mean(losses, axis=0)))

            if self.decay_rate:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                v_mrr, v_hit10, v_hit3, v_hit1 = self.evaluate(model, d.valid_data, W)
                print('Epoch %d valid, MRR=%.8f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (it, v_mrr, v_hit10, v_hit3, v_hit1))
                t_mrr, t_hit10, t_hit3, t_hit1 = self.evaluate(model, d.test_data, W)
                if v_mrr > best_valid_metric['mrr']:
                    best_valid_iter = it
                    print('======== MRR on validation set increases ======== ')
                    best_valid_metric['mrr'] = v_mrr
                    best_valid_metric['test_mrr'] = t_mrr
                    best_valid_metric['test_hit1'] = t_hit1
                    best_valid_metric['test_hit3'] = t_hit3
                    best_valid_metric['test_hit10'] = t_hit10
                else:
                    print('====Current Epoch:%d, Best Epoch:%d, valid_MRR didn\'t increase for %d Epoch, best test_MRR=%f' % (
                        it, best_valid_iter, it - best_valid_iter, best_valid_metric['test_mrr']))
                print('Epoch %d test, MRR=%.8f, Hits@10=%f, Hits@3=%f, Hits@1=%f' % (it, t_mrr, t_hit10, t_hit3, t_hit1))

                if  it == self.num_iterations: #(it-best_valid_iter) >= 10 or
                    print('++++++++++++ Early Stopping +++++++++++++')
                    print('Best epoch %d' % best_valid_iter)
                    print('Mean reciprocal rank: {0}'.format(best_valid_metric['test_mrr']))
                    print('Hits @10: {0}'.format(best_valid_metric['test_hit10']))
                    print('Hits @3: {0}'.format(best_valid_metric['test_hit3']))
                    print('Hits @1: {0}'.format(best_valid_metric['test_hit1']))
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WikiPeople-4", nargs="?", help="Which dataset to use: WikiPeople-3/4 or JF17K-3/4.")
    parser.add_argument("--num_iterations", type=int, default=200, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.006701566797680926, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--edim", type=int, default=25, nargs="?", help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=25, nargs="?", help="Relation embedding dimensionality.")
    parser.add_argument("--input_dropout", type=float, default=0.46694419227220374, nargs="?", help="Input layer dropout.")
    parser.add_argument("--hidden_dropout", type=float, default=0.18148844341064124, nargs="?", help="Hidden layer dropout.")
    parser.add_argument("--k", type=int, default=5, nargs="?", help="Reshaped tensor order")
    #parser.add_argument("--n_i", type=int, default=25, nargs="?", help="Mode-2 dimension of TR latent tensors.")
    parser.add_argument('--ni_list', nargs='+', type=int, default=[25,25,25,25,25], 
                    help='List of ni values per dimension (length=k)')
    parser.add_argument('--rank_list', nargs='+', type=int, default=[25,25,25,25,25,25], 
                    help='List of rank values per dimension (length=k+1)')
    parser.add_argument("--TR_ranks", type=int, default=40, nargs="?", help="TR-ranks")
    parser.add_argument("--method", choices=['TR', 'HT', 'TT', 'HT_ENHANCED', 'FC'], default='TR', help="Tensor decomposition method (TR or HT)")
    parser.add_argument("--ht_tree", type=str, choices=["A","C"], default="A",
                    help="HT tree config for arity=3: 'A' merges (e1,e2),(e3,rel) first; 'C' merges (e2,e3) first, then e1, then rel.")
    parser.add_argument("--ht4_tree", type=str, choices=["A", "B", "C"], default="A",
                    help="HT tree config for arity=3: 'A' merges (e1,e2),(e3,rel) first; 'C' merges (e2,e3) first, then e1, then rel.")
    parser.add_argument('--ht_ranks', nargs=3, type=int, required=True,
                    help="HT ranks r1,r2,r3 for the HT3_Enhanced model")
    parser.add_argument(
        '--mlp_hidden',
        type=int,
        default=None,
        help="(Optional) Hidden size of the small 2-layer MLP in HT_ENHANCED"
    )
    
    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "./data/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    d = Data(data_dir=data_dir)
    # print("Sample train_data[0]:", d.train_data[0])
    # print("Number of columns in train_data[0]:", len(d.train_data[0]))
    # print("Dataset should be 5 columns for WikiPeople-4 (relation, e1, e2, e3, e4)")

    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr,
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim,
                            k=args.k, ni=args.ni_list, ranks=args.TR_ranks,
                            input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout)
    experiment.train_and_eval()
