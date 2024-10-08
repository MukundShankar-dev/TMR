import numpy as np


def print_latex_metrics(metrics):
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    t2m_keys = [f"t2m/R{i}" for i in vals] + ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals] + ["m2t/MedR"]

    keys = t2m_keys + m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    print("Number of samples: {}".format(int(metrics["t2m/len"])))
    print(str_)


def all_contrastive_metrics(
    sims, keyids, emb=None, threshold=None, rounding=2, return_cols=False
):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_keyid_metrics, t2m_cols, top_t2m_retrievals = contrastive_metrics(
        sims, keyids, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_keyid_metrics, m2t_cols, top_m2t_retrievals = contrastive_metrics(
        sims.T, keyids, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    keyid_metrics = {"t2m": t2m_keyid_metrics, "m2t": m2t_keyid_metrics}
    retrieved = {"t2m_retrievals": top_t2m_retrievals, "m2t_retrievals": top_m2t_retrievals}

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols, keyid_metrics
    return all_m, keyid_metrics, top_t2m_retrievals, top_m2t_retrievals


def contrastive_metrics(
    sims,
    keyids,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    top_retrieval_indices = np.argmin(dists, axis=1)

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    metrics, keyid_metrics = cols2metrics(cols, num_queries,keyids, rounding=rounding)
    if return_cols:
        return metrics, keyid_metrics, cols, top_retrieval_indices
    return metrics, keyid_metrics


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries, keyids, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    all_keyid_metrics = {}
    for idx, keyid in enumerate(keyids):
        keyid_metrics = {"R01": False, "R02": False, "R03": False, "R05": False, "R10": False}
        keyid_val = cols[idx]
        if keyid_val < 1:
            keyid_metrics["R01"] = True
        elif keyid_val < 2:
            keyid_metrics["R02"] = True
        elif keyid_val < 3:
            keyid_metrics["R03"] = True
        elif keyid_val < 5:
            keyid_metrics["R05"] = True
        elif keyid_val < 10:
            keyid_metrics["R10"] = True
        all_keyid_metrics[keyid] = keyid_metrics

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics, all_keyid_metrics
