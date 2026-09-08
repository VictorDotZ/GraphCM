import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
import torch.nn.utils.rnn as rnn_utils
from torch_geometric.nn import GATConv
from torch_geometric.data import NeighborSampler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from torch_geometric.sampler import NeighborSampler as NeighborSamplerNew
from torch_geometric.sampler import NodeSamplerInput

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")
INF = 1e30
MINF = -(1e30)


# NOTE: The feature interaction module should have been implemented in RelEstimator logically.
#       For the convenience of implementation, we implement it in GNN Layers,
#       as the embeddings are saved in GNN Layers.
class DGATLayer(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(DGATLayer, self).__init__()
        self.args = args
        self.logger = logging.getLogger("GraphCM")
        self.query_size = query_size
        self.doc_size = doc_size
        self.vtype_size = vtype_size
        self.dataset = args.dataset
        self.data_dir = os.path.join("data", self.dataset)

        if args.use_pretrain_embed:
            self.qid_embedding = torch.load(
                os.path.join(self.data_dir, "embeddings/dgat_qid_embedding.pth")
            )
            self.uid_embedding = torch.load(
                os.path.join(self.data_dir, "embeddings/dgat_uid_embedding.pth")
            )
            assert self.qid_embedding.weight.data.shape[0] == query_size
            assert self.uid_embedding.weight.data.shape[0] == doc_size
            assert self.qid_embedding.weight.data.shape[1] == self.args.embed_size
            assert self.uid_embedding.weight.data.shape[1] == self.args.embed_size
        else:
            self.qid_embedding = nn.Embedding(query_size, self.args.embed_size)
            self.uid_embedding = nn.Embedding(doc_size, self.args.embed_size)
        self.click_embedding = nn.Embedding(2, self.args.click_embed_size)
        self.vid_embedding = nn.Embedding(vtype_size, self.args.vtype_embed_size)
        self.pos_embedding = nn.Embedding(10, self.args.pos_embed_size)

        if args.use_gnn:
            self.qid_edge_index = torch.load(
                os.path.join(self.data_dir, "dgat_qid_edge_index.pth")
            )
            self.qid_edge_index_train = torch.load(
                "/home/user/dev/thesis-click-models/train_q_q.pth",
                weights_only=False,
            )
            #
            self.qid_edge_index_test = torch.load(
                "/home/user/dev/thesis-click-models/test_q_q.pth",
                weights_only=False,
            )

            self.uid_edge_index = torch.load(
                os.path.join(self.data_dir, "dgat_uid_edge_index.pth")
            )
            if use_cuda:
                (
                    self.qid_edge_index,
                    self.qid_edge_index_train,
                    self.qid_edge_index_test,
                    self.uid_edge_index,
                ) = (
                    self.qid_edge_index.cuda(),
                    self.qid_edge_index_train.cpu(),
                    self.qid_edge_index_test.cpu(),
                    self.uid_edge_index.cuda(),
                )
            out_channel = (
                self.args.embed_size // self.args.gnn_att_heads
                if self.args.gnn_concat
                else self.args.embed_size
            )

            self.query_sampler = NeighborSampler(
                self.qid_edge_index,
                node_idx=None,
                sizes=[self.args.gnn_neigh_sample],
                batch_size=self.args.batch_size,
                return_e_id=False,
                shuffle=True,
                num_workers=12,
            )

            self.query_sampler_train = NeighborSamplerNew(
                self.qid_edge_index_train,
                num_neighbors=[self.args.gnn_neigh_sample],
                weight_attr="weight",
            )

            self.query_sampler_test = NeighborSamplerNew(
                self.qid_edge_index_test,
                num_neighbors=[self.args.gnn_neigh_sample],
                weight_attr="weight",
            )

            self.qid_GAT = GATConv(
                self.args.embed_size,
                out_channel,
                heads=self.args.gnn_att_heads,
                concat=self.args.gnn_concat,
                negative_slope=self.args.gnn_leaky_slope,
                dropout=self.args.gnn_dropout,
            )
            self.uid_GAT = GATConv(
                self.args.embed_size,
                out_channel,
                heads=self.args.gnn_att_heads,
                concat=self.args.gnn_concat,
                negative_slope=self.args.gnn_leaky_slope,
                dropout=self.args.gnn_dropout,
            )

        if self.args.inter_neigh_sample > 0:
            self.uid_neighbors = torch.load(
                os.path.join(self.data_dir, "dgat_uid_neighbors.pth")
            )
            self.interact_attention = nn.Linear(self.args.embed_size * 2, 1)
            self.interact_activation = nn.LeakyReLU(
                negative_slope=self.args.inter_leaky_slope
            )

    def neighbor_query_embedding(self, qids):
        qids = torch.tensor(qids).squeeze(-1)
        qids = qids.cpu()
        self.query_sampler = (
            self.query_sampler_train if self.training else self.query_sampler_test
        )

        # sampler_data = self.query_sampler.sample(qids)
        sampler_output = self.query_sampler.sample_from_nodes(
            NodeSamplerInput(
                qids,
                qids,
            )
        )

        n_id = sampler_output.node
        edge_index = torch.stack([sampler_output.row, sampler_output.col])

        # _, n_id, edge_index = sampler_data

        # print(n_id)
        # print(edge_index)
        # edge_index = edge_index.edge_index
        # raise Exception

        n_id, edge_index = n_id.cuda(), edge_index.cuda()
        x_all = self.qid_embedding(n_id)

        x_all = F.relu(self.qid_GAT(x_all, edge_index))
        # print(x_all, x_all.shape)

        return x_all[: qids.shape[0]]

    def forward(self, qids, uids, vids, clicks, use_gnn=True):
        # print("qids: ", qids)

        # Get click/vid/position embeddings
        CLICKS = rnn_utils.pad_sequence(
            [
                torch.from_numpy(np.array(click, dtype=np.int64))[:-1]
                for click in clicks
            ],
            batch_first=True,
        )
        VIDS = rnn_utils.pad_sequence(
            [torch.from_numpy(np.array(vid, dtype=np.int64)) for vid in vids],
            batch_first=True,
        )
        if use_cuda:
            CLICKS, VIDS = CLICKS.cuda(), VIDS.cuda()
        batch_size = CLICKS.shape[0]
        seq_len = CLICKS.shape[1]
        click_embedding = self.click_embedding(
            CLICKS
        )  # [batch_size, seq_len, click_embed_size]
        vid_embedding = self.vid_embedding(
            VIDS
        )  # [batch_size, seq_len, vtype_embed_size]
        pos_embedding = self.pos_embedding.weight.unsqueeze(dim=0).repeat(
            batch_size, seq_len // 10, 1
        )  # [batch_size, seq_len, embed_size]

        # Get qid/uid embeddings
        if use_gnn:
            # qid_neighbor_sampler = NeighborLoader(
            #     (
            #         self.qid_edge_index_train
            #         if self.training
            #         else self.qid_edge_index_test
            #     ),
            #     sizes=[self.args.gnn_neigh_sample],
            #     batch_size=self.query_size,
            #     weight_attr="weight",
            #     shuffle=True,
            #     num_workers=12,
            # )
            #
            # Ниже старый (китайский) вариант
            # qid_neighbor_sampler = NeighborSampler(
            #     self.qid_edge_index,
            #     node_idx=None,
            #     sizes=[self.args.gnn_neigh_sample],
            #     batch_size=self.query_size,
            #     return_e_id=False,
            #     shuffle=True,
            # )
            uid_neighbor_sampler = NeighborSampler(
                self.uid_edge_index,
                node_idx=None,
                sizes=[self.args.gnn_neigh_sample],
                batch_size=self.doc_size,
                return_e_id=False,
                shuffle=True,
                num_workers=12,
            )
            cnt = 0
            # for _, sampled_qid, sampled_index_tuple in qid_neighbor_sampler:
            #     # print(_)
            #     # print(sampled_qid)
            #     # print(sampled_index_tuple)
            #     # raise Exception
            #     assert cnt < 1
            #     cnt += 1
            #     if use_cuda:
            #         sampled_qid, sampled_index = (
            #             sampled_qid.cuda(),
            #             sampled_index_tuple[0].cuda(),
            #         )
            #     sampled_qid_embed = self.qid_embedding(sampled_qid)
            #     processed_qid_embed = F.relu(
            #         self.qid_GAT(sampled_qid_embed, sampled_index).type(torch.float)
            #     )
            #     argsort_sampled_qid = torch.argsort(sampled_qid)
            #     # Внутри этой хуйни лежат нужные нам обработанные эмбеды.
            #     # Наша задача -- понять, под каким индексом лежит эмбед внутри этой хуйни
            #     # который относится к нужным нам (переданным) куидам
            #     # print(
            #     #     "embedding after GAT: ",
            #     #     processed_qid_embed,
            #     #     processed_qid_embed.shape,
            #     # )
            #     # я не понимаю почему размерность этой штуки 165250.
            #     # Это же семплированные айдишники, тут должно быть только то, что попало в семпл
            #     # а у семпла размер 10
            #     # Если уменьшить batch_size до 4, то я увидел размерности 8, 12, 14.
            #     # Похоже что это просто количество соседей, которое может быть разным
            #     # print("sampled_qid: ", sampled_qid, sampled_qid.shape)
            #     # print(
            #     #     "argsort_sampled_qid: ",
            #     #     argsort_sampled_qid,
            #     #     argsort_sampled_qid.shape,
            #     # )
            # cnt = 0
            for _, sampled_uid, sampled_index_tuple in uid_neighbor_sampler:
                assert cnt < 1
                cnt += 1
                if use_cuda:
                    sampled_uid, sampled_index = (
                        sampled_uid.cuda(),
                        sampled_index_tuple[0].cuda(),
                    )
                sampled_uid_embed = self.uid_embedding(sampled_uid)
                processed_uid_embed = F.relu(
                    self.uid_GAT(sampled_uid_embed, sampled_index).type(torch.float)
                )
                argsort_sampled_uid = torch.argsort(sampled_uid)
            # Это куиды которые передали и зачем-то отшейпили их. По идее это оригинальные куиды
            # QIDS = rnn_utils.pad_sequence(
            #     [torch.from_numpy(np.array(qid, dtype=np.int64)) for qid in qids],
            #     batch_first=True,
            # )

            UIDS = rnn_utils.pad_sequence(
                [torch.from_numpy(np.array(uid, dtype=np.int64)) for uid in uids],
                batch_first=True,
            )
            # print("gnn, QIDS: ", QIDS, QIDS.shape)
            if use_cuda:
                UIDS = UIDS.cuda()

            # print("argsort_sampled_qid: ", argsort_sampled_qid)
            # что делает это чудо?
            # argsort_sampled_qid должен быть матрицей весов, откуда будут подставлятся эмбеды
            # processed_qid_embed это по идее обработанные эмбеды соседей, т.е
            # взяли запрос, взяли его соседей, посчитали эмбеды, свернули в один,
            # но т.к. китайцы, то вытащили слишком много и теперь нужен один.
            # wtf = F.embedding(argsort_sampled_qid, QIDS)
            # print("gnn, inner embed: ", wtf, wtf.shape)

            # qid_embedding = torch.gather(
            #     processed_qid_embed,
            #     dim=0,
            #     index=argsort_sampled_qid.unsqueeze(-1).expand(
            #         -1, self.args.embed_size
            #     ),
            # )[: self.args.batch_size].unsqueeze(1)

            qid_embedding = self.neighbor_query_embedding(qids=qids)
            qid_embedding = qid_embedding.unsqueeze(1)

            # print("qid_embedding: ", qid_embedding.shape)

            uid_embedding = torch.gather(
                processed_uid_embed,
                dim=0,
                index=argsort_sampled_uid.unsqueeze(-1).expand(
                    -1, self.args.embed_size
                ),
            )[UIDS.view(-1)].view(UIDS.shape[0], 10, self.args.embed_size)

            # print("uid_embedding: ", uid_embedding.shape)
        else:
            QIDS = rnn_utils.pad_sequence(
                [torch.from_numpy(np.array(qid, dtype=np.int64)) for qid in qids],
                batch_first=True,
            )
            UIDS = rnn_utils.pad_sequence(
                [torch.from_numpy(np.array(uid, dtype=np.int64)) for uid in uids],
                batch_first=True,
            )
            print("no gnn, QIDS: ", QIDS, QIDS.shape)
            if use_cuda:
                QIDS, UIDS = QIDS.cuda(), UIDS.cuda()
            qid_embedding = self.qid_embedding(QIDS)

            print(
                "no gnn, qid_embedding: ",
                qid_embedding,
                qid_embedding.shape,
            )

            uid_embedding = self.uid_embedding(UIDS)
            # print("uid_embedding: ", uid_embedding.shape)
        # raise Exception

        return (
            qid_embedding,
            uid_embedding,
            vid_embedding,
            click_embedding,
            pos_embedding,
        )

    def interact_neighs(self, qids, uids):
        batch_size = len(uids)
        seq_len = len(uids[0])
        QIDS = rnn_utils.pad_sequence(
            [torch.from_numpy(np.array(qid, dtype=np.int64)) for qid in qids],
            batch_first=True,
        )
        UIDS = rnn_utils.pad_sequence(
            [torch.from_numpy(np.array(uid, dtype=np.int64)) for uid in uids],
            batch_first=True,
        )
        if use_cuda:
            QIDS, UIDS = QIDS.cuda(), UIDS.cuda()
        batch_size = UIDS.shape[0]
        seq_len = UIDS.shape[1]

        qids_extended = (
            QIDS.unsqueeze(dim=2).repeat(1, 1, 10).view(batch_size, seq_len)
        )  # [batch_size, seq_len]
        qids_extended = qids_extended.unsqueeze(dim=2).repeat(
            1, 1, self.args.inter_neigh_sample
        )  # [batch_size, seq_len, inter_neigh_sample]
        qids_embed = self.qid_embedding(
            qids_extended
        )  # [batch_size, seq_len, inter_neigh_sample, embed_size]

        uids_perm_idx = torch.randperm(
            self.uid_neighbors.weight.data.shape[1], device=device
        )
        uids_neigh_idx = self.uid_neighbors(UIDS)[
            :, :, uids_perm_idx[: self.args.inter_neigh_sample]
        ]  # [batch_size, seq_len, inter_neigh_sample]
        uids_neigh = self.uid_embedding(
            uids_neigh_idx.to(torch.int64)
        )  # [batch_size, seq_len, inter_neigh_sample, embed_size]

        qu_interactions = qids_embed.mul(
            uids_neigh
        )  # [batch_size, seq_len, inter_neigh_sample, embed_size]

        attention_weights = torch.cat(
            [qids_embed, uids_neigh], dim=3
        )  # [batch_size, seq_len, inter_neigh_sample, embed_size * 2]
        attention_weights = self.interact_attention(attention_weights).squeeze(
            dim=3
        )  # [batch_size, seq_len, inter_neigh_sample]
        attention_weights = torch.exp(
            self.interact_activation(attention_weights)
        )  # [batch_size, seq_len, inter_neigh_sample]
        attention_weights = attention_weights / attention_weights.sum(dim=2).unsqueeze(
            dim=2
        )  # [batch_size, seq_len, inter_neigh_sample]

        qu_interactions = qu_interactions.mul(
            attention_weights.unsqueeze(dim=3)
        )  # [batch_size, seq_len, inter_neigh_sample, embed_size]
        qu_interactions = qu_interactions.sum(
            dim=2
        )  # [batch_size, seq_len, embed_size]

        return qu_interactions  # [batch_size, seq_len, embed_size]


class ExamPredictor(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(ExamPredictor, self).__init__()
        self.args = args
        self.logger = logging.getLogger("GraphCM")

        self.exam_gru = nn.GRU(
            self.args.pos_embed_size
            + self.args.vtype_embed_size
            + self.args.click_embed_size,
            self.args.hidden_size,
            batch_first=True,
        )
        self.exam_out_dim = self.args.hidden_size
        self.exam_output_linear = nn.Linear(self.exam_out_dim, 1)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, vid_embed, click_embed, pos_embed):
        batch_size = vid_embed.shape[0]
        seq_len = vid_embed.shape[1]
        exam_input = torch.cat((vid_embed, click_embed, pos_embed), dim=2)
        exam_state = Variable(
            torch.zeros(1, batch_size, self.args.hidden_size, device=device)
        )
        exam_outputs, exam_state = self.exam_gru(exam_input, exam_state)
        exam_outputs = self.dropout(exam_outputs)
        exams = self.sigmoid(self.exam_output_linear(exam_outputs)).view(
            batch_size, seq_len
        )
        return exams


class QueryEncoder(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(QueryEncoder, self).__init__()
        self.args = args
        self.logger = logging.getLogger("GraphCM")

        self.query_gru = nn.GRU(
            self.args.embed_size, self.args.hidden_size, batch_first=True
        )
        self.query_linear = nn.Linear(self.args.hidden_size, self.args.embed_size)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.activation = nn.Sigmoid()

    def forward(self, qid_embed):
        batch_size = qid_embed.shape[0]
        session_num = qid_embed.shape[1]
        query_state = Variable(
            torch.zeros(1, batch_size, self.args.hidden_size, device=device)
        )

        # print("qid_embed: ", qid_embed.shape)

        query_outputs, query_state = self.query_gru(
            qid_embed, query_state
        )  # [batch_size, session_num, hidden_size]
        query_outputs = query_outputs.repeat(1, 1, 10).view(
            batch_size, 10 * session_num, self.args.hidden_size
        )  # [batch_size, seq_len, hidden_size]
        query_outputs = self.dropout(query_outputs)
        encoded_query = self.activation(self.query_linear(query_outputs))

        return encoded_query  # [batch_size, seq_len, embed_size]


class DocEncoder(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(DocEncoder, self).__init__()
        self.args = args
        self.logger = logging.getLogger("GraphCM")

        self.doc_gru = nn.GRU(
            self.args.embed_size
            + self.args.pos_embed_size
            + self.args.vtype_embed_size
            + self.args.click_embed_size,
            self.args.hidden_size,
            batch_first=True,
        )
        self.doc_linear = nn.Linear(self.args.hidden_size, self.args.embed_size)
        self.dropout = nn.Dropout(p=self.args.dropout_rate)
        self.activation = nn.Sigmoid()

    def forward(self, uid_embed, vid_embed, click_embed, pos_embed):
        batch_size = uid_embed.shape[0]
        doc_input = torch.cat((uid_embed, vid_embed, click_embed, pos_embed), dim=2)
        doc_state = Variable(
            torch.zeros(1, batch_size, self.args.hidden_size, device=device)
        )
        doc_outputs, doc_state = self.doc_gru(doc_input, doc_state)
        doc_outputs = self.dropout(doc_outputs)
        encoded_doc = self.activation(self.doc_linear(doc_outputs))
        return encoded_doc  # [batch_size, seq_len, embed_size]


class RelEstimator(nn.Module):
    def __init__(self, args, query_size, doc_size, vtype_size, dataset):
        super(RelEstimator, self).__init__()
        self.args = args
        self.logger = logging.getLogger("GraphCM")

        self.query_encoder = QueryEncoder(
            args, query_size, doc_size, vtype_size, dataset
        )
        self.doc_encoder = DocEncoder(args, query_size, doc_size, vtype_size, dataset)
        mlp_input_dim = (
            self.args.embed_size * 3
            if self.args.inter_neigh_sample > 0
            else self.args.embed_size * 2
        )
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, self.args.embed_size),
            nn.Tanh(),
            nn.Linear(self.args.embed_size, 1),
            nn.Sigmoid(),
        )
        self.linear = nn.Linear(self.args.hidden_size * 2 + self.args.embed_size, 1)

    def forward(
        self, qid_embed, uid_embed, vid_embed, click_embed, pos_embed, qu_interactions
    ):
        batch_size = uid_embed.shape[0]
        seq_len = uid_embed.shape[1]
        encoded_query = self.query_encoder(qid_embed)
        encoded_doc = self.doc_encoder(uid_embed, vid_embed, click_embed, pos_embed)
        if qu_interactions is not None:
            mlp_input = torch.cat([encoded_query, encoded_doc, qu_interactions], dim=2)
        else:
            mlp_input = torch.cat([encoded_query, encoded_doc], dim=2)
        rels = self.mlp(mlp_input).view(batch_size, seq_len)
        return rels
