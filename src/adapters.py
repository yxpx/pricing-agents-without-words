"""Pinned local GPU adapters. No generation, paid API, or hidden agent memory.

predict accepts Clef-compatible records and returns probabilities in each record's
canonical sorted option identifiers A..K regardless of schema insertion order. Both adapters preserve the same economic state
and option descriptions; model-specific serialization/readout differ.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

REVISIONS = {
    "Cloudflare/clef-flash": "f0b7d3cd65296abfe459e6c228b618f5d8b26d2f",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def _validate(rows):
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("Invalid price probabilities")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError("Price probabilities do not sum to one")
    return arr / arr.sum(axis=1, keepdims=True)


class Adapter:
    def __init__(self, model_id="Cloudflare/clef-flash", revision=None,
                 batch_size=16, device="cuda", dtype="bfloat16",
                 max_tokens=2048, attention="sdpa", cache_dir=None):
        import torch
        from huggingface_hub import snapshot_download
        self.torch = torch
        self.model_id = model_id
        self.revision = revision or REVISIONS.get(model_id)
        if not self.revision:
            raise ValueError("An exact revision is required for an unlisted model")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.max_tokens = int(max_tokens)
        started = time.monotonic()
        self.path = snapshot_download(model_id, revision=self.revision, cache_dir=cache_dir)
        if model_id == "Cloudflare/clef-flash":
            # Use only the explicitly pinned, inspected custom implementation.
            spec = importlib.util.spec_from_file_location(
                "apart_pinned_clef_schema", Path(self.path) / "joint_schema_model.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.schema = module
            self.model, self.processor = module.load_release_model(
                self.path, device=self.device, dtype=self.dtype,
                attn_implementation=attention)
            self.tokenizer = self.processor.tokenizer
            self.kind = "native_joint_schema_head"
        else:
            from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.path, dtype=self.dtype, device_map={"": str(self.device)},
                attn_implementation=attention).eval()
            self.model.config.use_cache = False
            self.kind = "restricted_next_token_readout"
        self.model.eval()
        self.last_token_lengths = []
        self.metadata = {
            "model_id": model_id, "revision": self.revision, "kind": self.kind,
            "dtype": dtype, "attention": attention, "max_tokens": self.max_tokens,
            "batch_size": self.batch_size, "loading_seconds": time.monotonic()-started,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "packages": {p: importlib.metadata.version(p) for p in
                         ["torch", "transformers", "huggingface-hub", "safetensors", "numpy"]},
        }

    def _qwen_prompt(self, record):
        question = record["questions"]["price"]
        options = question["criteria"]
        for label in options:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Option {label!r} is not a single token")
        state = record["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        messages = [
            {"role": "system", "content": "Read the complete state and allowed options. Select exactly one option. Output only its identifier, without reasoning."},
            {"role": "user", "content": state + "\n\n" + question.get("instructions", "Choose a price.") + "\nAllowed options:\n" +
             "\n".join(f"{k}: {v}" for k,v in options.items())},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _predict_chunk(self, records):
        torch = self.torch
        if self.kind == "native_joint_schema_head":
            # Native encoder silently truncates to max_length. Encode with a roomy
            # bound, then reject rather than lose any economic observation.
            encoded = [self.schema.encode_record(self.tokenizer, r, max_length=65536,
                       processor=self.processor) for r in records]
            lengths = [len(e.input_ids) for e in encoded]
            if max(lengths) > self.max_tokens:
                raise ValueError(f"Input exceeds frozen token bound: {max(lengths)} > {self.max_tokens}")
            self.last_token_lengths.extend(lengths)
            batch = self.schema.collate_records(encoded, self.tokenizer.pad_token_id, self.device)
            with torch.inference_mode():
                logits = self.model(batch)
            rows = []
            for r, e, output in zip(records, encoded, logits):
                if len(e.questions) != 1 or e.questions[0].question_id != "price":
                    raise ValueError("Expected exactly the price question")
                prob = output[0].float().softmax(-1).cpu().numpy()
                by_id = dict(zip(e.questions[0].option_ids, prob))
                rows.append([by_id[k] for k in sorted(r["questions"]["price"]["criteria"])])
            return _validate(rows)

        prompts = [self._qwen_prompt(r) for r in records]
        batch = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        lengths = batch["attention_mask"].sum(1).tolist()
        if max(lengths) > self.max_tokens:
            raise ValueError(f"Input exceeds frozen token bound: {max(lengths)} > {self.max_tokens}")
        self.last_token_lengths.extend(lengths)
        # Check labels are one-token continuations in their exact prompt context.
        label_ids = []
        for r, prompt in zip(records, prompts):
            prefix = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = []
            for label in sorted(r["questions"]["price"]["criteria"]):
                continuation = self.tokenizer.encode(prompt+label, add_special_tokens=False)
                if continuation[:-1] != prefix:
                    raise ValueError("Option tokenization changes the prompt prefix")
                ids.append(continuation[-1])
            label_ids.append(ids)
        batch = {k:v.to(self.device) for k,v in batch.items()}
        text_model = self.model.model
        if hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        with torch.inference_mode():
            position_ids = batch['attention_mask'].long().cumsum(-1) - 1
            position_ids.masked_fill_(batch['attention_mask'] == 0, 0)
            out = text_model(**batch, position_ids=position_ids, use_cache=False, return_dict=True)
            # Left-padding means the final position is the last real token.
            last = out.last_hidden_state[:, -1, :]
            heads = self.model.get_output_embeddings()
            ids = torch.tensor(label_ids, device=self.device)
            weights = heads.weight[ids].float()
            scores = (weights * last.float()[:, None, :]).sum(-1)
            if getattr(heads, "bias", None) is not None:
                scores = scores + heads.bias[ids].float()
            rows = scores.softmax(-1).cpu().numpy()
        return _validate(rows)

    def predict(self, records):
        if not records:
            return np.empty((0, 11), dtype=np.float64)
        self.last_token_lengths = []
        rows = []
        # No stochastic model state; every call is a fresh full-context forward.
        for i in range(0, len(records), self.batch_size):
            rows.append(self._predict_chunk(records[i:i+self.batch_size]))
        return np.concatenate(rows, axis=0)

    def benchmark(self, records, batch_sizes=(8,16,32), repeats=2, max_seconds=180):
        torch = self.torch
        start = time.monotonic()
        results = []
        self.predict(records[:1])
        for size in batch_sizes:
            if time.monotonic()-start > max_seconds:
                break
            expanded = [records[i % len(records)] for i in range(size)]
            self.batch_size = size
            times=[]
            try:
                for _ in range(repeats):
                    if self.device.type == "cuda": torch.cuda.synchronize()
                    t=time.monotonic()
                    self.predict(expanded)
                    if self.device.type == "cuda": torch.cuda.synchronize()
                    times.append(time.monotonic()-t)
                results.append({"batch_size":size,"seconds":times,
                    "decisions_per_second":size/float(np.median(times)),
                    "tokens_min":min(self.last_token_lengths),"tokens_max":max(self.last_token_lengths)})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results.append({"batch_size":size,"error":"cuda_oom"})
                break
        valid=[r for r in results if "decisions_per_second" in r]
        if not valid: raise RuntimeError("No successful benchmark batch")
        best=max(valid,key=lambda r:r["decisions_per_second"])
        self.batch_size=best["batch_size"]
        self.metadata["batch_size"]=self.batch_size
        self.metadata["benchmark"]=results
        return results

    def check_batch_invariance(self, records, atol=0.01):
        """Check masking at unequal sequence lengths before scientific runs.

        BF16 batched GEMMs may differ slightly from singleton GEMMs. Retain the
        actual discrepancy; fail when differences exceed declared tolerance.
        This tests batches, not whether swapping allowed-option order is inert.
        """
        if len(records) < 2:
            raise ValueError("Supply at least two records of different lengths")
        saved_size = self.batch_size
        try:
            self.batch_size = 1
            singles = self.predict(records[:2])
            self.batch_size = 2
            together = self.predict(records[:2])
            error = float(np.max(np.abs(singles-together)))
            result = {"maximum_absolute_probability_difference":error,
                      "tolerance":atol,"passed":error <= atol,
                      "token_lengths":list(self.last_token_lengths)}
            self.metadata["batch_invariance"] = result
            if error > atol:
                raise ValueError(f"Batched/singleton probability difference {error} exceeds {atol}")
            return result
        finally:
            self.batch_size = saved_size


def create_adapter(**kwargs):
    return Adapter(**kwargs)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="Cloudflare/clef-flash")
    parser.add_argument("--records",required=True,help="JSON list of realistic benchmark records")
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    adapter=Adapter(model_id=args.model)
    adapter.benchmark(json.loads(Path(args.records).read_text()))
    Path(args.out).write_text(json.dumps(adapter.metadata,indent=2))
