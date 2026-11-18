"""Give a DFlash-style drafter (DFlash2, DSpine) its own LM head over the N most frequent tokens.

The drafter normally reuses the target's full output head (248,320 rows for Qwen3.8), which every draft pass reads:
~1.35 GB per GPU at Q8_0. This copies the target's Q8_0 rows for the N most frequent tokens into the drafter GGUF as
`output.weight`, plus `d2t` (draft row -> target token id). The patched llama.cpp scatters the draft logits back to the
full vocabulary. Drafts get cheaper; outputs are unchanged, because the target still verifies every token.

Step 1, token frequencies from your own text (any mix of prompts and model outputs), with llama-server running:
    python make_draft_head.py freq --server http://127.0.0.1:8080 --out freq.json corpus1.txt corpus2.txt ...
Step 2, the reduced-head drafter:
    python make_draft_head.py head --freq freq.json --n 65536 --draft drafter-Q4_K_M.gguf --target Qwen3.8-27B-Q8_0.gguf

Needs `gguf` from llama.cpp's gguf-py (pip install gguf, or put llama.cpp/gguf-py on PYTHONPATH) and numpy.
"""
import argparse
import collections
import json
import urllib.request

import numpy as np


def cmd_freq(args):
    cnt = collections.Counter()
    for path in args.files:
        text = open(path, encoding="utf-8", errors="replace").read()
        body = json.dumps({"content": text, "add_special": False, "parse_special": True}).encode()
        req = urllib.request.Request(args.server.rstrip("/") + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        cnt.update(json.loads(urllib.request.urlopen(req, timeout=600).read())["tokens"])
        print(f"{path}: {sum(cnt.values())} tokens so far, {len(cnt)} distinct")
    json.dump({"ranked": [t for t, _ in cnt.most_common()]}, open(args.out, "w"))
    print("wrote", args.out)


def cmd_head(args):
    import gguf

    ranked = json.load(open(args.freq))["ranked"]
    sel = list(dict.fromkeys(ranked[: args.n]))
    have = set(sel)
    i = 0
    while len(sel) < args.n:  # fill up with the lowest ids (special and byte tokens)
        if i not in have:
            sel.append(i)
            have.add(i)
        i += 1
    ids = np.array(sorted(sel), dtype=np.int64)

    src = gguf.GGUFReader(args.draft)
    tgt = gguf.GGUFReader(args.target)
    head = next(t for t in tgt.tensors if t.name == "output.weight")
    if head.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        raise SystemExit(f"target output.weight is {head.tensor_type.name}, expected Q8_0")
    if any(t.name in ("output.weight", "d2t") for t in src.tensors):
        raise SystemExit("the drafter already has its own head")
    rows = np.ascontiguousarray(head.data[ids])  # Q8_0 rows copied byte for byte

    out = args.out or args.draft[: -len(".gguf")] + f"-v{args.n // 1024}k.gguf"
    arch = src.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(out, arch=arch, endianess=src.endianess)
    for field in src.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
    for t in src.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.add_tensor_info("output.weight", rows.shape, rows.dtype, rows.nbytes, gguf.GGMLQuantizationType.Q8_0)
    w.add_tensor_info("d2t", ids.shape, ids.dtype, ids.nbytes, gguf.GGMLQuantizationType.I64)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in src.tensors:
        w.write_tensor_data(t.data, tensor_endianess=src.endianess)
    w.write_tensor_data(rows, tensor_endianess=src.endianess)
    w.write_tensor_data(ids, tensor_endianess=src.endianess)
    w.close()
    print("wrote", out, "with", len(ids), "head rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freq", help="count token frequencies through llama-server /tokenize")
    f.add_argument("--server", default="http://127.0.0.1:8080")
    f.add_argument("--out", default="freq.json")
    f.add_argument("files", nargs="+")
    h = sub.add_parser("head", help="write the drafter with an N-token head")
    h.add_argument("--freq", required=True)
    h.add_argument("--n", type=int, default=65536)
    h.add_argument("--draft", required=True)
    h.add_argument("--target", required=True)
    h.add_argument("--out")
    args = ap.parse_args()
    {"freq": cmd_freq, "head": cmd_head}[args.cmd](args)


if __name__ == "__main__":
    main()
