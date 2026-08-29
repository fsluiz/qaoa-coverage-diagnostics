"""Verify MDS logic for 3x3 grid Rydberg computation."""
import numpy as np

# 3x3 grid positions
delta = 5.0
R_adj = 6.2
n = 9
pos = np.array([[float(i*delta), float(j*delta)] for i in range(3) for j in range(3)])
diffs = pos[:, None, :] - pos[None, :, :]
dists = np.sqrt((diffs**2).sum(axis=2))
adj = ((dists > 0) & (dists <= R_adj)).astype(np.float64)

print("Adjacency matrix (3x3 grid, R=6.2µm):")
print(adj.astype(int))
print()

print("Node positions:")
for i in range(9):
    print(f"  Node {i}: ({pos[i,0]:.0f},{pos[i,1]:.0f}) µm, neighbors={np.where(adj[i])[0].tolist()}")

# Check known MDS: {2, 3, 7}
def is_mds(selected_nodes, adj):
    bits = np.zeros(9, dtype=float)
    for n_ in selected_nodes:
        bits[n_] = 1.0
    dominated = bits + adj @ bits
    return (dominated >= 1.0).all()

print()
test_cases = [
    ([2, 3, 7], "known MDS from rydberg_ds_results.json"),
    ([0, 2, 6, 8], "four corners (101000101)"),
    ([1, 3, 5, 7], "four edges"),
    ([0, 4, 8], "diagonal"),
    ([2, 7], "2 nodes"),
    ([4], "only center"),
    ([1, 3, 5], "3 edge nodes"),
    ([0, 4, 2, 6, 8], "center + corners"),
]
for nodes, desc in test_cases:
    result = is_mds(nodes, adj)
    print(f"  {nodes} ({desc}): MDS={result}")

print()
# Count how many of the 512 states are valid MDS
dim = 512
bit_shifts = np.arange(8, -1, -1, dtype=np.uint32)
idx_all = np.arange(dim, dtype=np.uint32)
bits_all = ((idx_all[:, None] >> bit_shifts[None, :]) & 1).astype(np.float64)
dominated = bits_all + bits_all @ adj.T
is_mds_arr = (dominated >= 1.0).all(axis=1)
print(f"Total MDS states: {is_mds_arr.sum()} out of 512")
k_mds = bits_all[is_mds_arr].sum(axis=1)
from collections import Counter
print(f"k distribution of MDS states: {dict(Counter(k_mds.astype(int).tolist()))}")

# Find all k=3 MDS states
k3_mds_idx = np.where(is_mds_arr & (bits_all.sum(axis=1) == 3))[0]
print(f"\nk=3 MDS states ({len(k3_mds_idx)} total):")
for s in k3_mds_idx:
    b = bits_all[s].astype(int)
    print(f"  |{''.join(map(str,b))}> = {np.where(b)[0].tolist()}")
