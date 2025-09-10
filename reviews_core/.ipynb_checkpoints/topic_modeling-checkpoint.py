import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path
import json, joblib


def apply_ldas(
    df: pd.DataFrame,
    text_col: str = "review_text_clean",
    # generic model (trained on all reviews)
    lda_path: str = "../assets/lda_final_ALL.pkl",
    vect_path: str = "../assets/vectorizer_final_ALL.pkl",
    # negative-only model
    lda_neg_path: str = "../assets/lda_final_NEG.pkl",
    vect_neg_path: str = "../assets/vectorizer_final_NEG.pkl",
    # positive-only model
    lda_pos_path: str = "../assets/lda_final_POS.pkl",
    vect_pos_path: str = "../assets/vectorizer_final_POS.pkl",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Append 9 columns using three LDA models:
      Generic: topic_id_ALL, topic_label_ALL, topic_prob_ALL
      Negative-only (score in {1,2}): topic_id_NEG, topic_label_NEG, topic_prob_NEG
      Positive-only (score in {4,5}): topic_id_POS, topic_label_POS, topic_prob_POS
    """
    if not inplace:
        df = df.copy()

    print(f"Initial shape of the dataframe: {df.shape}")

    # --- Text cleaning ----------------------------------------------------
    def text_cleaner(x):
        if not isinstance(x, str):
            return pd.NA
        t = x
        t = re.sub(r"@\w+", " ", t)                  # remove @mentions
        t = re.sub(r"\d+", " <num> ", t)             # numbers -> <num>
        t = re.sub(r"[^a-z\s<>]+", " ", t)           # keep letters, spaces, and <> so <num> survives
        t = re.sub(r"\s+", " ", t).strip()           # normalize whitespace
        return t if t else pd.NA

    df["review_extra_clean"] = df[text_col].map(text_cleaner)   

    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found.")

    s = df[text_col].astype("string")  # NA-safe string Series

    # ----------------- helper -----------------
    def _apply_one(lda, vect, mask_rows, label_map, col_suffix, nan_outside_mask: bool):
        """
        lda, vect: fitted artifacts
        mask_rows: boolean mask of rows to consider for this model
        label_map: dict[int,str] mapping topic id -> label
        col_suffix: '_ALL', '_NEG', or '_POS'
        nan_outside_mask:
            - True  -> rows outside mask get NaN in all 3 columns (for neg/pos models)
            - False -> rows outside eligibility get Undefined/-1/0.0 (for generic model)
        """
        if not hasattr(vect, "transform"):
            raise TypeError("Loaded vectorizer has no .transform()")

        # Eligibility rules
        analyzer = vect.build_analyzer()
        
        sw_ok = s.fillna("").map(lambda t: len(analyzer(t)) >= 1)
        consistent_ok = df["Consistent"].fillna(0).astype(int).eq(1)

        elig = mask_rows & sw_ok & consistent_ok

        # Prepare outputs
        if nan_outside_mask:
            # For neg/pos models we keep NaN outside the score mask
            topic_id = pd.Series(np.nan, index=df.index, dtype="float")
            topic_prob = pd.Series(np.nan, index=df.index, dtype="float")
            topic_label = pd.Series(np.nan, index=df.index, dtype="string")
        else:
            # For generic model, fill with Undefined defaults when ineligible
            topic_id = pd.Series(-1, index=df.index, dtype="int64")
            topic_prob = pd.Series(0.0, index=df.index, dtype="float64")
            topic_label = pd.Series("Undefined", index=df.index, dtype="string")

        # Vectorize eligible subset
        if elig.any():
            X = vect.transform(s[elig].fillna(""))

            # Rule 3: drop zero-feature rows (OOV)
            zero = np.array(X.sum(axis=1)).ravel() == 0
            if zero.any():
                bad_idx = s[elig].index[zero]
                elig.loc[bad_idx] = False
                if elig.any():
                    X = vect.transform(s[elig].fillna(""))

            # Score eligible
            if elig.any():
                D = lda.transform(X)                       # (n_docs_elig, n_topics)
                ti = D.argmax(axis=1)
                tp = D.max(axis=1)

                topic_id.loc[elig] = ti
                topic_prob.loc[elig] = tp
                topic_label.loc[elig] = pd.Series(ti, index=df[elig].index).map(
                    lambda k: label_map.get(int(k), "Undefined")
                ).astype("string")

        # Attach to df
        df[f"topic_id{col_suffix}"] = topic_id
        df[f"topic_label{col_suffix}"] = topic_label
        df[f"topic_prob{col_suffix}"] = topic_prob

    # ----------------- load artifacts -----------------
    lda_all = joblib.load(lda_path)
    vect_all = joblib.load(vect_path)
    lda_neg = joblib.load(lda_neg_path)
    vect_neg = joblib.load(vect_neg_path)
    lda_pos = joblib.load(lda_pos_path)
    vect_pos = joblib.load(vect_pos_path)

    # ----------------- label maps (edit to your final names) -----------------
    labels_gen = {
        0: "Fees & Service & Travel",
        1: "Updates, Crashes & Performance",
        2: "Login, Navigation & Functionality Issues",
        3: "Money Transfers & Payment Management",
        4: "Support & Access",
        5: "Transactions, Features & Usability",
        6: "Cards & Payments",
    } 
    labels_neg = {
        0: "Login & Security",
        1: "Payments & Cards",
        2: "App Updates & Crashes",
        3: "Transfers & Transaction",
        4: "Interface, Features & Fees",
        5: "Customer Service & Support",
    }
    # Stub/default for positive model – adjust to your pos-only training labels
    labels_pos = {
        0: "Customer Service & Usability",
        1: "Features, Security, Transactions",
        2: "Payments, Transfers & Money Mgmt",
        3: "Investments",
        4: "Cards & Travel",
        5: "Updates & Support",
    }

    # ----------------- run all three -----------------
    # Generic: run for everyone; ineligible -> Undefined/-1/0.0
    _apply_one(lda_all, vect_all, mask_rows=pd.Series(True, index=df.index), label_map=labels_gen,
               col_suffix="_ALL", nan_outside_mask=False)

    # Negative-only: only when score ∈ {1,2}; outside mask stays NaN
    _apply_one(lda_neg, vect_neg, mask_rows=df["score"].isin([1, 2]), label_map=labels_neg,
               col_suffix="_NEG", nan_outside_mask=True)

    # Positive-only: only when score ∈ {4,5}; outside mask stays NaN
    _apply_one(lda_pos, vect_pos, mask_rows=df["score"].isin([4, 5]), label_map=labels_pos,
               col_suffix="_POS", nan_outside_mask=True)

    # Order columns: keep original order, then append all new columns
    tail = [
        "topic_id_ALL", "topic_label_ALL", "topic_prob_ALL",
        "topic_id_NEG", "topic_label_NEG", "topic_prob_NEG",
        "topic_id_POS", "topic_label_POS", "topic_prob_POS",
    ]
    base = [c for c in df.columns if c not in tail]
    df = df[base + tail]

    print(f"Final shape of the dataframe: {df.shape}")

    # --- Save Final Dataframe ---
    save_path = "../assets/df_final.parquet"
    df.to_parquet(save_path, index = False)
    print(f"✅ Dataframe saved in {save_path}")

    return df