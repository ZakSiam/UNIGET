import torch
from torch_geometric.data import Data

from src.data.tokenizer.canonical import canonicalize_graph
from src.data.tokenizer.strategies.task_prep.pretrain import PretrainNTPStrategy
from src.data.tokenizer.types import TokenizationOutput


def make_graph():
    return Data(
        x=torch.tensor([[6], [8], [6], [7]], dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 1], [1, 0, 2, 1, 3, 2, 1, 3]],
            dtype=torch.long,
        ),
        edge_attr=torch.tensor([[1], [1], [2], [2], [3], [3], [4], [4]], dtype=torch.long),
        num_nodes=4,
    )


def relabel(graph, permutation):
    inv = torch.empty_like(permutation)
    inv[permutation] = torch.arange(permutation.numel())
    relabeled = Data(
        x=graph.x[permutation],
        edge_index=inv[graph.edge_index],
        edge_attr=graph.edge_attr.clone(),
        num_nodes=graph.num_nodes,
    )
    return relabeled


def canonical_signature(graph):
    result = canonicalize_graph(graph)
    rank = {node: idx for idx, node in enumerate(result.order)}
    return tuple((rank[src], rank[dst]) for src, dst in result.real_edges), tuple(
        (rank[src], rank[dst]) for src, dst in result.jump_edges
    )


def test_canonical_path_is_permutation_invariant_and_edge_once():
    graph = make_graph()
    permuted = relabel(graph, torch.tensor([2, 0, 3, 1]))
    assert canonical_signature(graph) == canonical_signature(permuted)

    result = canonicalize_graph(graph)
    undirected_real_edges = {tuple(sorted(edge)) for edge in result.real_edges}
    assert len(undirected_real_edges) == 4
    assert len(result.real_edges) == len(undirected_real_edges)


def test_ntp_strategy_keeps_shifted_labels_and_packed_metadata():
    strategy = PretrainNTPStrategy()
    in_dict = {
        "input_ids": [10, 11, 12],
        "labels": [11, -100, 2],
        "attention_mask": [1, 1, 1],
    }
    token_res = TokenizationOutput(ls_len=[2, 4])

    class DummyPacker:
        mpe = 8

    class DummyTokenizer:
        sequence_packer = DummyPacker()
        label_pad_token_id = -100

        @staticmethod
        def get_eos_token_id():
            return 2

    out = strategy.prepare(in_dict, token_res, graph=None, gtokenizer=DummyTokenizer())
    assert out["input_ids"] == [10, 11, 12, 2]
    assert out["labels"] == [11, -100, 2, -100]
    assert out["split_lens"] == [2, 2, 4]
    assert out["position_ids"] == [0, 1, 0, 1]
