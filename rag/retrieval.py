"""Retrieve neighboring triples and shortest paths for a question."""

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import torch
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from rag.core import (
    PathRetrievalError,
    format_path,
    masked_mean_pool,
    path_steps,
    unique_triples,
)


RETRIEVAL_PRESETS = {1: (4, 50, 30)}
SCORE_WEIGHTS = {
    1: (0.10, 0.05, 0.85),
    2: (0.05, 0.05, 0.90),
    3: (0.15, 0.05, 0.80),
}


class Neo4jHandler:

    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def find_shortest_path(self, start_node, end_node, max_hops=4):
        with self.driver.session(database=self.database) as session:
            result = session.execute_read(
                self._find_shortest_path, start_node, end_node, max_hops
            )
            return result

    @staticmethod
    def _find_shortest_path(tx, start_node, end_node, max_hops):
        query = f"""
            MATCH (start {{name: $start_node}}), (end {{name: $end_node}})
            MATCH p = shortestPath((start)-[*..{max_hops}]-(end))
            RETURN p
            """
        result = tx.run(
            query, start_node=start_node, end_node=end_node, max_hops=max_hops
        )
        paths = []
        for record in result:
            paths.append(path_steps(record['p']))
        return paths


class GraphRetriever:

    def __init__(self, args):
        self.args = args
        self.triple_csv_df = pd.read_csv(args.triple_csv_path)
        for col in ['Confidence_Score', 'Frequency_Score']:
            values = self.triple_csv_df[col]
            if (values == 'none').any() or values.isna().any():
                raise ValueError(f'{col} contains missing scores')
        self.tokenizer = AutoTokenizer.from_pretrained(args.biobert)
        self.bert_model = AutoModel.from_pretrained(args.biobert).eval()
        with np.load(args.ukb_node_emb, allow_pickle=True) as graph:
            self.graph_entities = graph['entities']
            self.graph_embeddings = graph['embeddings']
        self.entity_emb_npz = np.load(args.data_entity_emb, allow_pickle=True)
        self.neo4j_handler = Neo4jHandler(
            args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database
        )

    def close(self):
        try:
            self.neo4j_handler.close()
        finally:
            self.entity_emb_npz.close()

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embedding = masked_mean_pool(
            outputs.last_hidden_state, inputs['attention_mask']
        ).squeeze(0).numpy()
        return embedding

    def normalize(self, x):
        return (x + 1) / 2

    def get_nodes(self, question, qid: str, k1: int) -> List[str]:
        """Retrieve graph nodes for question entities, or for the whole question."""
        if qid not in self.entity_emb_npz:
            print(f'{qid} not in entity_emb_npz')
            q_emb = self.get_embedding(question)
            sims = cosine_similarity([q_emb], self.graph_embeddings)[0]
            top_idx = sims.argsort()[-k1:][::-1]
            return list(self.graph_entities[top_idx])
        entity_embeddings = self.entity_emb_npz[qid]
        nodes = set()
        for emb in entity_embeddings:
            sims = cosine_similarity([emb], self.graph_embeddings)[0]
            top_idx = sims.argsort()[-k1:][::-1]
            for idx in top_idx:
                nodes.add(self.graph_entities[idx])
        return list(nodes)

    def get_candidate_triples(self, entities: list) -> list:
        """Collect incoming/outgoing triples, keeping each directed triple once."""
        triples = []
        for entity in entities:
            head_df = self.triple_csv_df[self.triple_csv_df['HeadName'] == entity]
            tail_df = self.triple_csv_df[self.triple_csv_df['TailName'] == entity]
            for frame in (head_df, tail_df):
                for _, row in frame.iterrows():
                    triples.append([
                        row["HeadName"], row["RelName"], row["TailName"],
                        row["PMCID"], row["Confidence_Score"], row["Frequency_Score"],
                    ])
        return unique_triples(triples)

    def get_top_triples(self, question, triples, k2):

        def score(triple, q_emb):
            triple_str = f'[{triple[0]},{triple[1]},{triple[2]}]'
            t_emb = self.get_embedding(triple_str)
            sim = cosine_similarity([q_emb], [t_emb])[0][0]
            sim = self.normalize(sim)
            w1, w2, w3 = SCORE_WEIGHTS[self.args.score]
            confidence_score = triple[4]
            frequency_score = triple[5]
            combined_score = w1 * confidence_score + w2 * frequency_score + w3 * sim
            return combined_score
        q_emb = self.get_embedding(question)
        scored = [(t, score(t, q_emb)) for t in triples]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_triples = [t for (t, _) in scored[:k2]]
        return top_triples

    def get_top_paths(self, question, paths, k3):
        q_emb = self.get_embedding(question)
        scored = []
        for p in paths:
            p_emb = self.get_embedding(p)
            sim = cosine_similarity([q_emb], [p_emb])[0][0]
            sim = self.normalize(sim)
            scored.append((p, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_paths = [p for (p, _) in scored[:k3]]
        return top_paths

    def neighbor_based_retriever(self, triples, pmcids):
        """Render the ranked neighboring triples in the experiment format."""
        output_text = '\n\n## Neighbor-based Context:\n\nTriples:\n'
        for (idx, (triple, pmcid)) in enumerate(zip(triples, pmcids)):
            output_text += f'{idx + 1}. {triple}\n'
        return output_text.strip()

    def path_based_retriever(self, question, entities: list, k3) -> str:
        """Query node pairs and rank their paths by question similarity."""
        entities = list(set(entities))
        path_text_list = []
        for start, end in combinations(entities, 2):
            for path in self.neo4j_handler.find_shortest_path(start, end):
                text = format_path(path)
                if text:
                    path_text_list.append(text)
        top_paths = self.get_top_paths(question, path_text_list, k3)
        output_text = '\n\n## Path-based Context:\n\nPaths:\n'
        for (idx, path) in enumerate(top_paths):
            output_text += f'{idx + 1}. {path}\n'
        return output_text

    def retriever(self, qid, question: str):
        k1, k2, k3 = RETRIEVAL_PRESETS[self.args.params]
        nodes = self.get_nodes(question, qid, k1)
        candidate_triples = self.get_candidate_triples(nodes)
        top_triples_raw = self.get_top_triples(question, candidate_triples, k2)
        top_triples = [
            f'[{triple[0]} -({triple[1]})-> {triple[2]}]'
            for triple in top_triples_raw
        ]
        top_pmcids = [triple[3] for triple in top_triples_raw]
        context = self.neighbor_based_retriever(top_triples, top_pmcids)
        try:
            context += self.path_based_retriever(question, nodes, k3)
        except Exception as e:
            raise PathRetrievalError(
                f'Path retrieval failed ({type(e).__name__}); skipping question {qid}'
            ) from e
        return context
