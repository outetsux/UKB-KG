"""Extract question entities, map UMLS names and encode them with BioBERT."""

import argparse
import json
import os
from pathlib import Path

from rag.core import masked_mean_pool, read_json, select_entities, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", choices=["pubmedqa", "bioasq"], default="pubmedqa")
    parser.add_argument("--data-path")
    parser.add_argument("--cache-dir", help="Directory for entity and embedding files.")
    parser.add_argument("--llm", default="gpt-5")
    parser.add_argument("--effort", default="minimal")
    parser.add_argument("--spacy-model", default="en_core_sci_scibert")
    parser.add_argument("--umls-threshold", type=float, default=0.98)
    parser.add_argument("--biobert", default=os.environ.get("UKB_BIOBERT", "model/biobert_v1.2"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    root = Path("rag/data") / args.data
    args.data_path = args.data_path or str(root / "test_set.json")
    cache = Path(args.cache_dir) if args.cache_dir else root
    args.entity_path = cache / "test_set_entity.json"
    args.mapped_path = cache / "test_set_entity_umls_map.json"
    args.emb_path = cache / "test_set_entity_map_emb.npz"
    if Path(args.data_path).resolve() in {
        path.resolve() for path in [args.entity_path, args.mapped_path, args.emb_path]
    }:
        parser.error("Input data must be separate from generated files.")
    return args


def ner(text, nlp, chat, form, prompt_template, args):
    try:
        seen = set()
        mentions = []
        for ent in nlp(text).ents:
            if ent.text in seen:
                continue
            if any(token.pos_ in {"NOUN", "PROPN"} for token in ent):
                mentions.append(ent.text)
                seen.add(ent.text)
    except Exception as exc:
        print(f"NER error: {exc}")
        raise
    if not mentions:
        print("no entities found")
        return None
    try:
        prompt = prompt_template.replace("<<entities>>", str(mentions))
        response = chat(args.llm, prompt, form, args.effort)
        if not response["entities"]:
            print("no filtered entities found")
            return None
        return select_entities(mentions, response["entities"])
    except Exception as exc:
        print(f"LLM filter error: {exc}")
        raise


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import prompts
    import spacy
    import torch
    from pydantic import BaseModel
    from transformers import AutoModel, AutoTokenizer
    from llm.openai_chat import chat_structured
    from utils.umls import umls_map

    class EntityFilterForm(BaseModel):
        entities: list[str]

    with open(args.data_path, encoding="utf-8") as stream:
        data = json.load(stream)
    nlp = spacy.load(args.spacy_model)
    prompt_template = prompts.entity_filter_prompt_structured
    entities = read_json(args.entity_path)
    for qid, details in data.items():
        if qid in entities or not details.get("QUESTION"):
            continue
        extracted = ner(details["QUESTION"], nlp, chat_structured, EntityFilterForm, prompt_template, args)
        if extracted:
            entities[qid] = extracted
    write_json(args.entity_path, entities)

    mapper = umls_map(threshold=args.umls_threshold)
    mapped = read_json(args.mapped_path)
    for qid, mentions in entities.items():
        if qid not in mapped:
            mapped[qid] = [mapper(mention) for mention in mentions]
    write_json(args.mapped_path, mapped)

    tokenizer = AutoTokenizer.from_pretrained(args.biobert)
    model = AutoModel.from_pretrained(args.biobert)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    embeddings = {}
    if Path(args.emb_path).exists():
        with np.load(args.emb_path, allow_pickle=True) as cache:
            embeddings = dict(cache)
    for qid, names in mapped.items():
        if qid in embeddings:
            continue
        inputs = tokenizer(names, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings[qid] = masked_mean_pool(outputs.last_hidden_state, inputs["attention_mask"]).cpu().numpy()
    path = Path(args.emb_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **embeddings)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved {len(embeddings)} question embeddings to {path}")


if __name__ == "__main__":
    main()
