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
        offset of each marker inside the rendered string.

        Some templates reject a system role. Three renderings are tried in
        order and the one used is kept in `self.template_mode` and printed
        once, because a silent change of prompt format changes every result:
          system+user  -> the model's own chat template with a system turn
          user-only    -> chat template, system text prepended to the user turn
          plain        -> no chat template at all"""
        attempts = (("system+user", [{"role": "system", "content": spec.system}, {"role": "user", "content": spec.user}]),
                    ("user-only", [{"role": "user", "content": spec.system + "\n\n" + spec.user}]))
        text, mode = None, "plain"
        for name, msgs in attempts:
            try:
                cand = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                continue
            if cand and spec.user in cand and (name == "user-only" or spec.system in cand):
                text, mode = cand, name
                break
        if text is None:
            text = spec.system + "\n\n" + spec.user + "\n"
        if getattr(self, "template_mode", None) != mode:
            self.template_mode = mode
            print(f"[probe] prompt rendering: {mode}" + ("" if mode == "system+user" else
                  "  <-- NOT the standard system+user chat format; check this is intended"), flush=True)
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


    # ------------------------------------------------------------------
    # Block-level capture and patching (used by scripts/06_patching.py).
    #
    # These work on the residual stream *leaving transformer block b*
    # (b = 0..L-1), taken with forward hooks. For b < L-1 that equals
    # hidden_states[b+1]; for the last block it is the PRE-final-norm state,
    # which is what a patch must edit. Positions are addressed from the END
    # of the sequence, because a clause-present and a clause-replaced prompt
    # share an identical suffix (everything after the story) but not a
    # prefix length.
    def _layers(self):
        base = self.model.get_decoder() if hasattr(self.model, "get_decoder") else self.model.model
        for name in ("layers", "h"):
            if hasattr(base, name):
                return getattr(base, name)
        raise RuntimeError("could not locate the list of transformer blocks")

    def _encode(self, spec: PromptSpec):
        text, markers = self.render(spec)
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.pop("offset_mapping")[0].tolist()
        return enc["input_ids"].to(self.device), offsets, markers

    def suffix_len(self, spec: PromptSpec) -> int:
        """Number of tokens lying wholly after the story block."""
        ids, offsets, markers = self._encode(spec)
        cut = markers["after_story"]
        return sum(1 for (s, e) in offsets if s >= cut and e > s)

    def offset_from_end(self, spec: PromptSpec, marker: str) -> int:
        """How many positions from the end the marker's token sits (1 = the
        answer position). Cells that differ only in list order or only in the
        mapping share this offset, which is what cross-cell patching needs."""
        ids, offsets, markers = self._encode(spec)
        if marker == "answer":
            return 1
        return int(ids.shape[1] - self._token_index_for_char(offsets, markers[marker]))

    def describe_markers(self, spec: PromptSpec) -> Dict[str, Tuple[int, str]]:
        """marker -> (token index, token text). Print this once per model and
        read it: an off-by-one here silently moves every read-out."""
        ids, offsets, markers = self._encode(spec)
        out = {}
        for name, cpos in markers.items():
            i = self._token_index_for_char(offsets, cpos)
            out[name] = (i, self.tok.decode(ids[0, i: i + 1]))
        out["answer"] = (ids.shape[1] - 1, self.tok.decode(ids[0, -1:]))
        return out

    def _result(self, spec: PromptSpec, logits) -> CallResult:
        probs = self.torch.softmax(logits.float(), dim=-1)
        lm, cov = label_mass(probs, self.sets)
        gd = to_goal_distribution(lm, spec)
        return CallResult(spec.fmt, spec.order, spec.mapping, lm, cov, gd, max(gd, key=gd.get))

    def run_blocks(self, spec: PromptSpec, n_suffix: int, capture: Sequence[int] = (),
                   edit: Optional[Tuple[int, "object"]] = None):
        """One forward pass. `capture`: block indices whose output at the last
        n_suffix positions is returned as {block: (n_suffix, d) float32 cpu}.
        `edit` = (block, fn): fn maps that block's (n_suffix, d) float32 slice
        to its replacement, written back in the forward pass so every later
        block sees it. Returns (CallResult, captured)."""
        torch = self.torch
        ids, _, _ = self._encode(spec)
        if n_suffix > ids.shape[1]:
            raise ValueError("n_suffix longer than the sequence")
        layers = self._layers()
        grabbed: Dict[int, "object"] = {}
        handles = []

        def make_hook(b):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                seg = h[0, -n_suffix:, :].detach().float().cpu().clone()
                if b in capture:
                    grabbed[b] = seg
                if edit is not None and edit[0] == b:
                    new = edit[1](seg).to(device=h.device, dtype=h.dtype)
                    h = h.clone()
                    h[0, -n_suffix:, :] = new
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return None
            return hook

        for b in set(capture) | ({edit[0]} if edit is not None else set()):
            handles.append(layers[b].register_forward_hook(make_hook(b)))
        try:
            with torch.no_grad():
                out = self.model(ids)
        finally:
            for h in handles:
                h.remove()
        return self._result(spec, out.logits[0, -1]), grabbed

    def reference_vectors(self, texts: Sequence[str]):
        """Context-free representation of each text: hidden state at its last
        token, every layer -> (n_texts, L+1, d) float32. Used as the shared
        content space for the selection index (goal ids mean nothing across
        families; these vectors do)."""
        torch = self.torch
        out = []
        for t in texts:
            enc = self.tok(t, return_tensors="pt")
            with torch.no_grad():
                hs = self.model(enc["input_ids"].to(self.device), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1] for h in hs]).float().cpu())
        return torch.stack(out)


# ---------------------------------------------------- permutation-averaged run
def run_design(probe: OpenWeightsProbe, specs: Sequence[PromptSpec], **kw) -> Tuple[Dict[str, float], List[CallResult]]:
    """Run every spec in a design (4 native rotations, or 16 decoupled cells),
    return the averaged semantic distribution and the per-call results."""
    calls = [probe.call(s, **kw) for s in specs]
    return average_distributions([c.goal_dist for c in calls]), calls
