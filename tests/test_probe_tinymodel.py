"""Integration test for probe.py on a randomly initialised tiny Llama with a
tiny BPE tokenizer. Requires torch + transformers + tokenizers; skipped
otherwise. Checks: (1) label token sets merge surface forms, (2) label mass
and coverage come from the exact distribution, (3) the after_candidates
marker maps to a token before the answer mapping, (4) the final-layer logit
lens reproduces the model's logits — the regression test for the
double-normalisation bug in the original mech_readout.py."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from moriarty_geometry.prompts import build_decoupled, build_native, latin_orders, latin_mappings
from moriarty_geometry import probe as P


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=["[UNK]", "[PAD]"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["STORY OPENING: the story has not started yet. CANDIDATE HIDDEN GOALS: A. B. C. D. "
              "first second third fourth ANSWER MAPPING: -> Which is most likely? Answer with the "
              "single letter assigned to it. goal true distractor one two three " * 20]
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


class TinyProbe(P.OpenWeightsProbe):
    def __init__(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        self.torch = torch
        self.tok = tiny_tokenizer()
        cfg = LlamaConfig(vocab_size=len(self.tok), hidden_size=32, intermediate_size=64,
                          num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=512)
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(cfg).eval()
        self.device = "cpu"
        self.sets = P.label_token_sets(self.tok)
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.get_output_embeddings()
        self.capture_layers = None


def test_label_sets_mass_markers_and_lens_invariant():
    pr = TinyProbe()
    assert all(len(v) >= 1 for v in pr.sets.values()), pr.sets
    goals = ["true goal", "distractor one", "distractor two", "distractor three"]
    spec = build_decoupled(["the story has not started yet"], goals, latin_orders()[1], latin_mappings()[0])
    res = pr.call(spec, capture=("after_candidates", "answer"), logit_lens=True)
    assert abs(sum(res.letter_mass.values()) - 1) < 1e-6 and 0 <= res.coverage <= 1
    assert res.hidden["answer"].shape[0] == pr.n_layers + 1
    text, markers = pr.render(spec)
    assert markers["after_candidates"] < text.find("ANSWER MAPPING")
    # the final-layer invariant is asserted inside _logit_lens; reaching here means it held
    assert res.lens is not None and len(res.lens) == pr.n_layers + 1


def test_native_and_decoupled_agree_on_letter_semantics():
    goals = ["g0", "g1", "g2", "g3"]
    n = build_native([], goals, 2)
    d = build_decoupled([], goals, latin_orders()[2], ("A", "B", "C", "D"))
    assert n.letter_to_goal == d.letter_to_goal


def _pn_specs():
    goals = ["true goal", "distractor one", "distractor two", "distractor three"]
    o, m = latin_orders()[1], latin_mappings()[0]
    P_ = build_decoupled(["the story has not started yet goal one"], goals, o, m)
    N_ = build_decoupled(["the story has not started yet two three true"], goals, o, m)
    return P_, N_


def test_markers_point_at_candidate_ends_and_suffix_is_shared():
    pr = TinyProbe()
    P_, N_ = _pn_specs()
    d = pr.describe_markers(P_)
    idx = [d[f"cand_end_{k}"][0] for k in range(4)]
    assert idx == sorted(idx) and len(set(idx)) == 4
    assert d["cand_end_3"][0] == d["after_candidates"][0] < d["post_list"][0] < d["answer"][0]
    assert pr.offset_from_end(P_, "post_list") == pr.offset_from_end(N_, "post_list") > 1
    assert pr.suffix_len(P_) == pr.suffix_len(N_) > 0


def test_identity_patch_is_a_noop_and_last_block_patch_reproduces_donor():
    pr = TinyProbe()
    P_, N_ = _pn_specs()
    n = min(pr.suffix_len(P_), pr.suffix_len(N_))
    last = pr.n_layers - 1
    base_P, hP = pr.run_blocks(P_, n, capture=range(pr.n_layers))
    base_N, hN = pr.run_blocks(N_, n, capture=range(pr.n_layers))
    assert max(abs(base_P.goal_dist[g] - base_N.goal_dist[g]) for g in base_P.goal_dist) > 1e-6
    # identity patch
    same, _ = pr.run_blocks(P_, n, edit=(1, lambda h: h))
    assert all(abs(same.goal_dist[g] - base_P.goal_dist[g]) < 1e-5 for g in same.goal_dist)
    # replacing N's last-block output at the answer position by P's gives P's answer exactly
    def donor(h):
        h = h.clone(); h[-1] = hP[last][-1]; return h
    patched, _ = pr.run_blocks(N_, n, edit=(last, donor))
    assert all(abs(patched.goal_dist[g] - base_P.goal_dist[g]) < 1e-5 for g in patched.goal_dist)
    # block capture agrees with hidden_states for a non-final block
    res = pr.call(P_, capture=("answer",))
    assert torch.allclose(res.hidden["answer"][1].float(), hP[0][-1], atol=1e-5)


def test_reference_vectors_shape():
    pr = TinyProbe()
    R = pr.reference_vectors(["true goal", "distractor one"])
    assert R.shape[0] == 2 and R.shape[1] == pr.n_layers + 1
