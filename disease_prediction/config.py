"""Shared paths for KG preparation, classifier entry points and plotting.

Run from the repository root. Export SAVE_ROOT/RUN_NAME from the upstream
shell config, or override them with UKB_KG_ROOT/UKB_KG_RUN. Explicit Python
CLI paths take precedence over these defaults.
"""

import os

# --------------------------------------------------------------- KG artifacts
KG_ROOT = os.environ.get("UKB_KG_ROOT", os.environ.get("SAVE_ROOT", "save"))
KG_RUN = os.environ.get("UKB_KG_RUN", os.environ.get("RUN_NAME", "sample"))
KG_DIR = os.path.join(KG_ROOT, KG_RUN)

# Final triples, output of run_postprocess.sh stage 'csv'.
TRIPLES_JSON = os.path.join(KG_DIR, "merged", "10_triples.json")

# Final edge list with confidence scores, output of stage 'attributes'.
TRIPLES_CSV = os.path.join(KG_DIR, "csv", "triples_with_attributes.csv")

# Derived here, in 01_relation_mapping.py / 03_get_entity_emb.py.
MAPPED_TRIPLES_JSON = TRIPLES_JSON.replace(".json", "_mapped_12r.json")
RELATION_MAPPING_JSON = TRIPLES_JSON.replace(".json", "_mapping_12r.json")
MAPPED_TRIPLES_TXT = MAPPED_TRIPLES_JSON.replace(".json", ".txt")
NODE_EMB_NPZ = os.path.join(KG_DIR, "csv", "node_emb.npz")

# ------------------------------------------------------------- cohort  data
DATA_DIR = os.environ.get("UKB_DP_DATA", "disease_prediction/data")
RAW_DIR = os.path.join(DATA_DIR, "data_raw")

PHECODES_NPY = os.path.join(DATA_DIR, "phecodes.npy")
PHECODE_DEF_CSV = os.path.join(DATA_DIR, "phecode_def_abbr.csv")

# --------------------------------------------------------------- KGE toolkits
# The KGE trainers themselves are external; only their data/model directories
# are referenced here.
KGE_DIR = os.environ.get("UKB_KGE_DIR", "disease_prediction")
KGE_MODEL = os.environ.get("UKB_KGE_MODEL", "TransE")
KGE_SAVE_ID = os.environ.get("UKB_KGE_SAVE_ID", "0")
KGE_DATASET = os.environ.get(
    "UKB_KGE_DATASET",
    "ukb_gpt-5-minimal_12r" if KG_RUN == "sample" else f"ukb_{KG_RUN}_12r",
)
TAG = KGE_DATASET.removeprefix("ukb_")
PHECODE_NODE_MAP_CSV = os.path.join(
    DATA_DIR, "phecode_node_map.csv" if TAG == "gpt-5-minimal_12r"
    else f"phecode_node_map_{TAG}.csv",
)
SAVE_DIR = os.environ.get("UKB_DP_SAVE", "disease_prediction/save")


def kge_toolkit(model_name):
    if model_name not in ("TransE", "RotatE", "ComplEx", "HAKE", "ModE"):
        raise ValueError(f"Unsupported KGE model: {model_name}")
    return "KGE-HAKE" if model_name in ("HAKE", "ModE") else "KGE-RotatE"


KGE_DATA_DIR = os.path.join(KGE_DIR, kge_toolkit(KGE_MODEL), "data", KGE_DATASET)
ENTITIES_DICT = os.path.join(KGE_DATA_DIR, "entities.dict")
RELATIONS_DICT = os.path.join(KGE_DATA_DIR, "relations.dict")


def kge_entity_embedding(model_name):
    """Entity embedding matrix written by the KGE trainer for `model_name`."""
    return os.environ.get("UKB_KGE_EMBEDDING", os.path.join(
        KGE_DIR, kge_toolkit(model_name), "models",
        f"{model_name}_{KGE_DATASET}_{KGE_SAVE_ID}", "entity_embedding.npy",
    ))


def ene_output_dir(model_name):
    """Where 05_to_ene.py writes the per-subject KG-embedding features."""
    return os.path.join(DATA_DIR, f"data_{TAG}_{model_name}")


# ------------------------------------------------------------------- models
BIOBERT_PATH = os.environ.get("UKB_BIOBERT", "model/biobert_v1.2")


def classifier_dir(data_type, model_name=KGE_MODEL):
    name = "nn5_binary" if data_type == "binary" else f"nn5_{TAG}_{model_name}_ene"
    return os.path.join(SAVE_DIR, name)


if __name__ == "__main__":
    import shlex
    # Consumed by train.sh/test.sh; quoting also supports paths with spaces.
    for key, value in {
        "RAW_DIR": RAW_DIR, "TAG": TAG, "SAVE_ROOT_DP": SAVE_DIR,
        "DATA_DIR": ene_output_dir(KGE_MODEL),
        "SAVE_DIR": classifier_dir(os.environ.get("DATA_TYPE", "ene")),
    }.items():
        print(f"{key}={shlex.quote(value)}")
