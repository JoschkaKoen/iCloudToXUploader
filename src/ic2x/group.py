"""Cluster image DB rows into groups for multi-image tweets."""

from __future__ import annotations

# X / Twitter caps a single tweet at 4 attached images.
MAX_IMAGES_PER_TWEET = 4


def cluster_by_phash(rows: list, threshold: int) -> list[list]:
    """Greedy first-fit clustering by pHash Hamming distance.

    Each image joins the first existing group with a member within `threshold`
    distance and room under `MAX_IMAGES_PER_TWEET`.
    threshold=0 → no grouping; every image is its own group.
    """
    if threshold <= 0:
        return [[row] for row in rows]

    import imagehash

    groups: list[list] = []
    for row in rows:
        phash = row["phash"] or ""
        placed = False
        if phash:
            h = imagehash.hex_to_hash(phash)
            for group in groups:
                if len(group) >= MAX_IMAGES_PER_TWEET:
                    continue
                for member in group:
                    mphash = member["phash"] or ""
                    if mphash and (h - imagehash.hex_to_hash(mphash)) <= threshold:
                        group.append(row)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            groups.append([row])
    return groups
