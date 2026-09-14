"""Open-weights probe.

Fixes relative to the original mech_readout.py / probe_openweights.py:

* Label mass is computed from the *exact* full-vocabulary next-token
  distribution (no top-20 truncation). Coverage = mass on the four label
  token sets before renormalisation, recorded for every call.
* Hidden states are captured at named positions (marker char offsets in the
  user message mapped to token indices via the tokenizer's offset mapping),
  not only at the last token.
* Logit lens: HF `output_hidden_states` returns L+1 tensors; the last one has
  ALREADY passed through the model's final norm (Llama/Qwen2/Mistral all do
  `hidden_states = self.norm(hidden_states)` before appending). The original
  readout normalised every entry again, double-normalising the final layer.
  Here the final entry is unembedded directly and the reconstructed final
  distribution is asserted to match the model's own logits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .prompts import LETTERS, PromptSpec


# ------------------------------------------------------------------ label mass
def label_token_sets(tokenizer) -> Dict[str, List[int]]:
    """All single-token surface forms that whitespace-strip and case-fold to
    a label letter. Computed once per tokenizer from the full vocabulary so
    that ' A', 'A', 'a', ' a' etc. are all merged — the same rule as the
    original probe, applied exhaustively."""
    sets: Dict[str, List[int]] = {L: [] for L in LETTERS}
    vocab = tokenizer.get_vocab()
    for tok, idx in vocab.items():
        s = tokenizer.convert_tokens_to_string([tok]).strip().upper()
        if s in sets:
            sets[s].append(idx)
    return sets


def label_mass(probs, sets: Dict[str, List[int]]) -> Tuple[Dict[str, float], float]:
    """probs: 1-D tensor/array over the vocabulary (already softmaxed).
    Returns (renormalised mass per letter, coverage)."""
    raw = {L: float(sum(float(probs[i]) for i in ids)) for L, ids in sets.items()}
    cov = sum(raw.values())
    if cov <= 0:
        return {L: 0.0 for L in LETTERS}, 0.0
    return {L: v / cov for L, v in raw.items()}, cov


def to_goal_distribution(letter_mass: Dict[str, float], spec: PromptSpec) -> Dict[str, float]:
    return {spec.letter_to_goal[L]: letter_mass[L] for L in LETTERS}


def average_distributions(dists: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = dists[0].keys()
    return {k: sum(d[k] for d in dists) / len(dists) for k in keys}


# --------------------------------------------------------------- probe results
@dataclass
class CallResult:
    spec_fmt: str
    order: Tuple[int, ...]
    mapping: Tuple[str, ...]
    letter_mass: Dict[str, float]
    coverage: float
    goal_dist: Dict[str, float]
    argmax_goal: str
    hidden: Dict[str, "object"] = field(default_factory=dict)   # marker -> (L+1, d) tensor
    lens: Optional[List[Dict[str, float]]] = None               # per-layer goal dists


# --------------------------------------------------------------------- probe
class OpenWeightsProbe:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16",
                 capture_layers: Optional[Sequence[int]] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.device = device
        self.sets = label_token_sets(self.tok)
        base = self.model.get_decoder() if hasattr(self.model, "get_decoder") else self.model.model
        self.final_norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)
        self.lm_head = self.model.get_output_embeddings()
        if self.final_norm is None or self.lm_head is None:
            raise RuntimeError("could not locate final norm / lm_head")
        self.capture_layers = capture_layers

    # ---- prompt rendering with marker mapping ----
    def render(self, spec: PromptSpec) -> Tuple[str, Dict[str, int]]:
        """Apply the chat template; return the rendered string and the char
        offset of each marker inside the rendered string."""
        msgs = [{"role": "system", "content": spec.system}, {"role": "user", "content": spec.user}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = spec.system + "\n\n" + spec.user + "\n"
        u0 = text.find(spec.user)
        if u0 < 0:
            raise RuntimeError("user text not found verbatim in rendered template")
        markers = {name: u0 + off for name, off in spec.markers.items()}
        return text, markers

    def _token_index_for_char(self, offsets, char_pos: int) -> int:
        """Index of the last token that ends at or before char_pos."""
        idx = 0
        for i, (s, e) in enumerate(offsets):
            if e <= char_pos and e > 0:
                idx = i
        return idx

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    def call(self, spec: PromptSpec, capture: Iterable[str] = ("answer",),
             logit_lens: bool = False) -> CallResult:
        torch = self.torch
        text, markers = self.render(spec)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model(ids, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        lm, cov = label_mass(probs, self.sets)
        goal_dist = to_goal_distribution(lm, spec)
        res = CallResult(spec.fmt, spec.order, spec.mapping, lm, cov, goal_dist,
                         max(goal_dist, key=goal_dist.get))
        hs = out.hidden_states  # tuple of L+1, each (1, T, d)
        pos_of = {"answer": ids.shape[1] - 1}
        for name, cpos in markers.items():
            pos_of[name] = self._token_index_for_char(offsets, cpos)
        layers = list(self.capture_layers) if self.capture_layers is not None else list(range(len(hs)))
        for name in capture:
            p = pos_of[name]
            res.hidden[name] = torch.stack([hs[l][0, p] for l in layers]).to("cpu")
        if logit_lens:
            res.lens = self._logit_lens(hs, ids.shape[1] - 1, spec, logits)
        return res

    def _logit_lens(self, hs, pos: int, spec: PromptSpec, true_logits) -> List[Dict[str, float]]:
        torch = self.torch
        out = []
        last = len(hs) - 1
        for l, h in enumerate(hs):
            v = h[0, pos]
            if l < last:
                v = self.final_norm(v)            # intermediate layers: apply final norm (logit lens)
            v = v.to(self.lm_head.weight.dtype)  # final layer: already normed by the model
            lg = self.lm_head(v).float()
            if l == last:
                # invariant: reconstructed final-layer logits equal the model's logits
                if not torch.allclose(lg, true_logits, atol=1e-2, rtol=1e-2):
                    raise RuntimeError("final-layer lens does not reproduce model logits — "
                                       "check norm handling for this architecture")
            lm, _ = label_mass(torch.softmax(lg, -1), self.sets)
            out.append(to_goal_distribution(lm, spec))
        return out


# ---------------------------------------------------- permutation-averaged run
def run_design(probe: OpenWeightsProbe, specs: Sequence[PromptSpec], **kw) -> Tuple[Dict[str, float], List[CallResult]]:
    """Run every spec in a design (4 native rotations, or 16 decoupled cells),
    return the averaged semantic distribution and the per-call results."""
    calls = [probe.call(s, **kw) for s in specs]
    return average_distributions([c.goal_dist for c in calls]), calls
