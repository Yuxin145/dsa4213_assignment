"""Test 15 (generation_consistency) + matched decoding and text metrics."""
import numpy as np
import pytest
import torch

from src.generate import NeuralScorer, NgramScorer, choose_next, generate, text_metrics
from src.models import ARCHS, build_model
from src.ngram import InterpolatedTrigram


@pytest.mark.parametrize("arch", ARCHS)
def test_greedy_equals_argmax_of_full_forward(cfg, smoke_data, cpu, arch):
    vocab = smoke_data["vocab"]
    V = len(vocab)
    model = build_model(arch, cfg, V, 0, seed=0).eval()
    with torch.no_grad():  # make the output layer favour <pad> so that the pad mask is exercised
        model.out.bias[0] = 50.0
    prompt = vocab.encode(cfg["generation"]["prompts"][0].split())
    gen = generate(NeuralScorer(model, cpu, False, 128), prompt, 30, vocab.pad_id, "greedy")
    assert 0 not in gen, "<pad> was generated"
    seq = prompt + gen
    with torch.no_grad():
        for i in range(len(gen)):
            prefix = torch.tensor([seq[:len(prompt) + i]])
            logits = model(prefix)[0, -1]
            logits[0] = -float("inf")
            assert int(torch.argmax(logits)) == gen[i], f"step {i}"
        # a single full forward pass over the final sequence gives the same argmaxes (causality)
        logits = model(torch.tensor([seq]))[0]
        logits[:, 0] = -float("inf")
        pred = logits.argmax(-1)[len(prompt) - 1:-1].tolist()
    assert pred == gen


def test_ngram_greedy_and_sampling(smoke_data):
    vocab = smoke_data["vocab"]
    ng = InterpolatedTrigram(len(vocab), 0).fit(smoke_data["streams"]["train"])
    ng.set_lambdas((0.5, 0.3, 0.15, 0.05))
    prompt = vocab.encode("the history of the city".split())
    sc = NgramScorer(ng)
    gen = generate(sc, prompt, 20, 0, "greedy")
    seq = prompt + gen
    for i in range(len(gen)):
        d = ng.distribution(seq[:len(prompt) + i])
        d[0] = -1
        assert int(np.argmax(d)) == gen[i]
    s1 = generate(sc, prompt, 20, 0, "sample", 0.8, 40, seed=7)
    s2 = generate(sc, prompt, 20, 0, "sample", 0.8, 40, seed=7)
    assert s1 == s2 and 0 not in s1


def test_top_k_and_temperature():
    logp = torch.log_softmax(torch.tensor([5.0, 1.0, 2.0, 3.0, 0.5, 4.0], dtype=torch.float64), -1)
    g = torch.Generator().manual_seed(0)
    picks = {choose_next(logp, 0, "sample", 0.8, 2, g) for _ in range(300)}
    assert picks == {3, 5}, "top-k must keep exactly the k best non-pad tokens; pad never sampled"
    assert choose_next(logp, 0, "greedy") == 5


def test_text_metrics():
    m = text_metrics([["a", "b", "c", "a", "b", "c"], ["<unk>", "x"]])
    assert m["n_tokens"] == 8
    assert m["distinct_1"] == pytest.approx(5 / 8)
    assert m["distinct_2"] == pytest.approx(4 / 6)  # (a,b),(b,c),(c,a),(<unk>,x) unique of 6 bigrams
    assert m["repeated_trigram_rate"] == pytest.approx(1 / 4)  # (a,b,c) repeats once among 4 trigrams
    assert m["unk_rate"] == pytest.approx(1 / 8)
