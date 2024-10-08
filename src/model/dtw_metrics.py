import numpy as np

# Get all metrics - t2m, m2t for a given similarity matrix and protocol (determined when called)
def all_contrastive_metrics(
    sims, keyids, ref_df, dtw_scores, emb=None, threshold=None, rounding=2, return_cols=False
):
    # emb decides if it's protocol (b) or not (use of sentence embeddings)
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols, top_t2m_retrievals = contrastive_metrics(
        sims, keyids, ref_df, dtw_scores, text_selfsim=text_selfsim, threshold=threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols, top_m2t_retrievals = contrastive_metrics(
        sims.T, keyids, ref_df, dtw_scores, text_selfsim=text_selfsim, threshold=threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m, top_t2m_retrievals, top_m2t_retrievals

def contrastive_metrics(
    sims,
    keyids,
    ref_df,
    dtw_scores,
    text_selfsim=None,
    threshold=None,
    return_cols=False,
    rounding=2,
    break_ties="averaging",
):
    # Convert keyids to a np array for easier querying
    keyids = np.array(keyids)
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    top_retrieval_indices = np.argsort(dists, axis=1)[:, :10]

    # Go over what this does exactly
    if text_selfsim is not None and threshold is not None:
        # Get threshold to be between -1 and 1
        real_threshold = 2 * threshold - 1
        # Take where the text sim matrix is above this threshold
        idx = np.argwhere(text_selfsim > real_threshold)
        
        # From GPT:
        # The code finds unique row indices from idx and their first occurrences.
        # It then computes the minimum distances in dists for these unique rows using a partitioned reduction operation.
        # Finally, it reshapes the resulting array of minimum distances into a column vector.
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    # Check up what rows and cols are
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT
    
    # Write a test case for this
    for i, row in enumerate(top_retrieval_indices):
        row_keyids = keyids[row]
        mask = ref_df['keyids'].isin(row_keyids)
        matching_indices = ref_df.index[mask].tolist()
        row_dtw_scores = np.array(dtw_scores[i][matching_indices])
        try:
            best_col = np.min(np.where(row_dtw_scores < 300)[1])
            if best_col < cols[i]:
                cols[i] = best_col
        except:
            pass

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

    metrics = cols2metrics(cols, num_queries,keyids, rounding=rounding)
    if return_cols:
        return metrics, cols, top_retrieval_indices
    return metrics

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

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics
