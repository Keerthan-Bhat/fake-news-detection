# explain_utils.py

import numpy as np


def get_top_words_for_text(pipeline, text: str, top_k: int = 10):
    """
    For a given text, returns:
      - top words pushing towards Fake (class 1)
      - top words pushing towards Real (class 0)
    using TF-IDF * Logistic Regression coefficients.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X_vec = tfidf.transform([text])

    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]  # shape: (n_features,)

    # Contribution of each word = TF-IDF value * weight
    contrib = X_vec.toarray()[0] * coefs

    # Top positive = Fake
    top_fake_idx = np.argsort(contrib)[-top_k:][::-1]
    # Top negative = Real
    top_real_idx = np.argsort(contrib)[:top_k]

    top_fake_words = [
        (feature_names[i], float(contrib[i]))
        for i in top_fake_idx if contrib[i] > 0
    ]
    top_real_words = [
        (feature_names[i], float(contrib[i]))
        for i in top_real_idx if contrib[i] < 0
    ]

    return top_fake_words, top_real_words
