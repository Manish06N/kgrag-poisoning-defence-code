"""RAGDefender (Kim, Lee, Koo; arXiv 2511.01268) adapted to retrieved KG paths: each verbalised path plays the role of a passage and the
question's retrieved paths are its retrieved set. A removal RULE (not a thresholded score): per question, estimate the number of adversarial
items N_adv (stage 1), rank items by a similarity-concentration score f_i, remove the top N_adv (stage 2).

  stage 1, "conc"  (multi-hop, concentration-based grouping): s_mean_i / s_median_i = mean / median cosine similarity of item i to the others;
                   N_adv = #{i : s_mean_i > mean(s_mean) and s_median_i > median(s_median)}
  stage 1, "clust" (single-hop, clustering-based grouping): agglomerative clustering into 2 groups; TF-IDF top-m terms (m=5) of the set;
                   N_TFIDF = #{items containing more than 2 of the top-m terms}; N_adv = size of the smaller group if N_TFIDF <= n/2 else the larger
  stage 2:         N_pairs = max(1, N_adv // 2) most similar pairs; f_i = sum over those pairs containing i of sgn(sim) * |sim|^p (p = 2);
                   remove the N_adv items with the largest f_i.
Deviations from the paper (state them): sentence embeddings are all-MiniLM-L6-v2 (the paper uses Stella) over verbalised KG paths; the
paper's stage 1 also uses the query similarity distribution, which is not used here; at least one path is always kept; questions with
fewer than 3 paths are left untouched.

  python -m kgrag.defended_run --defence ragdefender:conc        # or ragdefender:clust
Sensitivity variant "oraclen": stage 2 is given the TRUE number of poisoned paths of each question (an upper bound for the ranking step,
not a deployable defence).
"""
import numpy as np

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    return _ENC


def verbalise(text: str) -> str:
    return text.replace(".", " ").replace("_", " ")


def estimate_conc(S: np.ndarray) -> int:
    n = len(S)
    off = ~np.eye(n, dtype=bool)
    mean = np.array([S[i][off[i]].mean() for i in range(n)])
    med = np.array([np.median(S[i][off[i]]) for i in range(n)])
    return int(((mean > mean.mean()) & (med > np.median(med))).sum())


def estimate_clust(emb: np.ndarray, texts: list[str], m: int = 5) -> int:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(texts)
    labels = AgglomerativeClustering(n_clusters=2).fit_predict(emb)
    sizes = np.bincount(labels, minlength=2)
    n_min, n_max = int(sizes.min()), int(sizes.max())
    try:
        vec = TfidfVectorizer()
        X = vec.fit_transform(texts)
        top = np.argsort(-np.asarray(X.sum(0)).ravel())[:m]
        n_tfidf = int(((X[:, top] > 0).sum(1) > 2).sum())
    except ValueError:  # empty vocabulary
        n_tfidf = 0
    return n_min if n_tfidf <= n / 2 else n_max


def question_removals(emb: np.ndarray, texts: list[str], variant: str = "conc", p: int = 2, n_adv: int | None = None) -> np.ndarray:
    """Boolean mask over one question's paths (True = remove). `n_adv` overrides the stage-1 estimate (used by the oracle sensitivity)."""
    n = len(texts)
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    S = emb @ emb.T
    if n_adv is None:
        n_adv = estimate_conc(S) if variant == "conc" else estimate_clust(emb, texts)
    n_adv = min(n_adv, n - 1)
    if n_adv <= 0:
        return out
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-S[iu])[: max(1, n_adv // 2)]
    f = np.zeros(n)
    for k in order:
        i, j = iu[0][k], iu[1][k]
        w = np.sign(S[i, j]) * abs(S[i, j]) ** p
        f[i] += w
        f[j] += w
    out[np.argsort(-f)[:n_adv]] = True
    return out


def removal_mask(recs, variant: str = "conc") -> np.ndarray:
    """Flat boolean removal mask over the paths of `recs` (same order as defend.flat)."""
    enc = _encoder()
    out = []
    for r in recs:
        texts = [verbalise(p["text"]) for p in r["paths"]]
        if len(texts) < 3:
            out.extend([False] * len(texts))
            continue
        emb = enc.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
        true_n = sum(bool(p["poisoned"]) for p in r["paths"]) if variant == "oraclen" else None
        out.extend(question_removals(emb, texts, variant, n_adv=true_n).tolist())
    return np.array(out, dtype=bool)
