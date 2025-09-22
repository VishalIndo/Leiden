# --- install if needed ---

%matplotlib inline
import numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ARI
from matplotlib import animation, colors
from IPython.display import HTML
import igraph as ig
import random
from queue import SimpleQueue
from math import exp
import leidenalg as la  # for Scanpy modularity partition

# =========================================================
# 0) Repro & plotting config
# =========================================================
np.random.seed(42)
random.seed(42)
sc.set_figure_params(figsize=(5,5), dpi=110)
sc.settings.verbosity = 2

# =========================================================
# 1) Load PBMC3k and Scanpy preprocessing
# =========================================================
adata = sc.datasets.pbmc3k()

# Light QC (tweak thresholds if needed)
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs['pct_counts_mt'] < 12].copy()
adata = adata[adata.obs['n_genes_by_counts'] < 2500].copy()

# Normalize, log, HVGs, scale (sparse-safe), PCA (sparse-safe)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.scale(adata, zero_center=False)
sc.tl.pca(adata, svd_solver="arpack", zero_center=False)

# Neighbor graph & UMAP
N_NEI, N_PCS = 15, 30
sc.pp.neighbors(adata, n_neighbors=N_NEI, n_pcs=N_PCS, use_rep="X_pca", metric="euclidean")
sc.tl.umap(adata)

# =========================================================
# 2) Scanpy Leiden (library) on modularity for fair comparison
# =========================================================
sc.tl.leiden(
    adata,
    key_added="leiden_lib",
    partition_type=la.ModularityVertexPartition,  # match your custom objective
    resolution=None,                               # modularity has no resolution
    flavor="leidenalg",
    directed=False,
    use_weights=True,
    n_iterations=-1,                               # run optimiser to convergence
)
print("Scanpy (library) leiden_lib clusters:", adata.obs["leiden_lib"].nunique())

# =========================================================
# 3) Convert Scanpy connectivities to igraph (same kNN graph)
# =========================================================
A = adata.obsp["connectivities"].tocoo()
edges = {}
for i, j, w in zip(A.row, A.col, A.data):
    if i == j:
        continue
    a, b = (int(i), int(j)) if i < j else (int(j), int(i))
    edges[(a, b)] = edges.get((a, b), 0.0) + float(w)

elist = [(u, v, w) for (u, v), w in edges.items()]
g = ig.Graph()
g.add_vertices(adata.n_obs)
g.add_edges([(u, v) for (u, v, _) in elist])
g.es["weight"] = [w for (_, _, w) in elist]
g.vs["name"] = list(adata.obs_names)

# =========================================================
# 4) From-scratch LEIDEN (full implementation, epsilon-safe)
# =========================================================
_comm = "leiden_commLeiden"
_refine = "leiden_commLeidenRefinement"
_refineIndex = "leiden_refinementIndex"
_queued = "leiden_queued"
_wellconnected = "leiden_wellconnected"
_multiplicity = "leiden_multiplicity"
_degree = "leiden_degree"
_selfEdges = "leiden_selfEdges"
_m = "leiden_m"
theta = 1e-2
EPS = 1e-9

def _clip0(x, eps=EPS): return 0.0 if abs(float(x)) < eps else float(x)
def quality(graph, comm): return graph.modularity(graph.vs[comm])

def initialiseGraph(graph):
    m2 = 0.0
    if graph.is_weighted():
        deg = []
        for v in range(graph.vcount()):
            s = 0.0
            for nb in graph.neighbors(v):
                s += graph[v, nb]
            s = _clip0(s)
            deg.append(s)
            graph.vs[v][_degree] = s
            m2 += s
        graph.vs[_degree] = deg
    else:
        graph.es["weight"] = 1.0
        graph.vs[_degree] = graph.degree(range(graph.vcount()))
        m2 = float(sum(graph.vs[_degree]))
    graph.vs[_selfEdges] = 0.0
    graph.vs[_multiplicity] = 1
    graph[_m] = _clip0(m2 / 2.0)

def initialisePartition(graph, attribute):
    communities = {}
    for idx, v in enumerate(graph.vs):
        v[attribute] = idx
        communities[idx] = (v[_multiplicity], 0.0, v[_degree], True)
    communities[-1] = (0, 0.0, 0.0, True)
    return communities

def cleanCommunities(communities):
    out = {}
    for key, val in communities.items():
        vcount, int_edges, degsum, flag = val
        int_edges = _clip0(int_edges); degsum = _clip0(degsum)
        if vcount != 0:
            out[key] = (vcount, int_edges, degsum, flag)
        else:
            if (abs(int_edges) >= EPS) or (abs(degsum) >= EPS):
                raise ValueError(f"Community with {int_edges} internal edges and {degsum} internal degree without internal vertices found")
    out[-1] = (0, 0.0, 0.0, True)
    return out

def makeQueue(n):
    q = SimpleQueue(); idx = list(range(n)); random.shuffle(idx)
    for i in idx: q.put(i)
    return q

def calculateDQPlus(graph, communities, comm, vertex, edges, degree):
    _, _, degsum, _ = communities[comm]; m = graph[_m]
    return edges / m - (2.0 * degsum * degree) / (2.0 * m) ** 2

def calculateDQMinus(graph, communities, comm, vertex, edges, degree):
    _, _, degsum, _ = communities[comm]; m = graph[_m]
    return -edges / m + (2.0 * (degsum - degree) * degree) / (2.0 * m) ** 2

def update_communities(communities, current, future, community_edges, multiplicity, self_edges, degree):
    old_vc, old_ec, old_deg, _ = communities[current]
    new_vc = old_vc - multiplicity
    new_ec = _clip0(old_ec - community_edges.get(current, self_edges))
    new_deg = _clip0(old_deg - degree)
    communities[current] = (new_vc, new_ec, new_deg, True)

    old_vc2, old_ec2, old_deg2, _ = communities.get(future, (0, 0.0, 0.0, True))
    new_vc2 = old_vc2 + multiplicity
    new_ec2 = _clip0(old_ec2 + community_edges.get(future, self_edges))
    new_deg2 = _clip0(old_deg2 + degree)
    communities[future] = (new_vc2, new_ec2, new_deg2, True)

def localMove(graph, communities):
    queue = makeQueue(graph.vcount())
    graph.vs[_queued] = True
    while not queue.empty():
        v = queue.get()
        degree = graph.vs[v][_degree]
        current_comm = graph.vs[v][_comm]
        self_edges = graph.vs[v][_selfEdges]
        neigh = graph.neighbors(v)

        community_edges = {-1: self_edges}
        for nb in neigh:
            comm = graph.vs[nb][_comm]
            community_edges[comm] = community_edges.get(comm, self_edges) + graph[v, nb]

        max_dq = 0.0; max_comm = current_comm
        cost_leaving = calculateDQMinus(graph, communities, current_comm, v, community_edges.get(current_comm, self_edges), degree)
        for comm, eweight in community_edges.items():
            if comm == current_comm: continue
            dq = calculateDQPlus(graph, communities, comm, v, eweight, degree) + cost_leaving
            if dq > max_dq: max_dq, max_comm = dq, comm

        if max_comm != current_comm:
            if max_comm == -1:
                i = 0
                while True:
                    if communities.get(i, (0, 0.0, 0.0, True))[0] == 0: break
                    i += 1
                max_comm = i
            graph.vs[v][_comm] = max_comm
            update_communities(communities, current_comm, max_comm, community_edges, graph.vs[v][_multiplicity], self_edges, degree)
            for nb in neigh:
                if (not graph.vs[nb][_queued]) and (graph.vs[nb][_comm] != max_comm):
                    graph.vs[nb][_queued] = True; queue.put(nb)
        graph.vs[v][_queued] = False

def refine(graph, communities, refine_communities, simplified):
    converged = True
    graph.vs[_queued] = True
    comms = list(communities.keys()); random.shuffle(comms)

    for comm in comms:
        sel_kw = {_comm + "_eq": comm}
        indices = [v.index for v in graph.vs.select(**sel_kw)]
        degreesum = communities[comm][2]; random.shuffle(indices)

        if not simplified:
            for v in indices:
                neigh_in = graph.vs[graph.neighbors(v)].select(**sel_kw)
                degree = graph.vs[v][_degree]; edges_sum = 0.0
                for nb in neigh_in: edges_sum += graph[v, nb.index]
                well = edges_sum >= degree * (degreesum - degree) / (graph[_m] * 2.0)
                graph.vs[v][_wellconnected] = bool(well)
                a, b, c, _ = refine_communities[graph.vs[v][_refine]]
                refine_communities[graph.vs[v][_refine]] = (a, b, c, bool(well))

        for v in indices:
            if not graph.vs[v][_queued]: continue
            if (not simplified) and (not graph.vs[v][_wellconnected]): continue

            neigh_in = graph.vs[graph.neighbors(v)].select(**sel_kw)
            degree = graph.vs[v][_degree]; current_ref = graph.vs[v][_refine]
            self_edges = graph.vs[v][_selfEdges]

            community_edges = {}; neighbor_rep = {}
            for nb in neigh_in:
                ref_comm = nb[_refine]
                if (not simplified) and (not refine_communities[ref_comm][3]): continue
                community_edges[ref_comm] = community_edges.get(ref_comm, self_edges) + graph[v, nb.index]
                neighbor_rep[ref_comm] = nb

            candidates, weights = [], []
            cost_leaving = calculateDQMinus(graph, refine_communities, current_ref, v, self_edges, degree)
            for rc, eweight in community_edges.items():
                dq = calculateDQPlus(graph, refine_communities, rc, v, eweight, degree) + cost_leaving
                if dq > 0: candidates.append(rc); weights.append(np.exp(dq / theta))
            if candidates:
                target = candidates[int(np.argmax(weights))] if simplified else random.choices(candidates, weights)[0]
                graph.vs[v][_refine] = target
                update_communities(refine_communities, current_ref, target, community_edges, self_edges=self_edges, multiplicity=graph.vs[v][_multiplicity], degree=degree)
                neighbor_rep[target][_queued] = False; graph.vs[v][_queued] = False; converged = False

                if not simplified:
                    kw_t = {_refine + "_eq": target}
                    members = [vv.index for vv in graph.vs[indices].select(**kw_t)]
                    e2 = 0.0
                    for m in members:
                        for nb in graph.vs[graph.neighbors(m)].select(**sel_kw):
                            e2 += graph[v, nb.index]
                    ref_mult, ref_edges, ref_deg, _ = refine_communities[target]
                    if not (e2 - 2 * ref_edges) >= ref_deg * (degreesum - ref_deg) / (graph[_m] * 2.0):
                        refine_communities[target] = (ref_mult, ref_edges, ref_deg, False)
    return converged

def aggregate(graph, communities):
    part = ig.VertexClustering.FromAttribute(graph, _refine)
    agg = part.cluster_graph({None: "first", _multiplicity: "sum", _degree: "sum"}, "sum")
    del part
    idx_map = {v[_refine]: v.index for v in agg.vs}
    graph.vs[_refineIndex] = [idx_map[ref] for ref in graph.vs[_refine]]
    agg.vs[_selfEdges] = [communities[v[_refine]][1] for v in agg.vs]
    return agg

def deAggregate(graph, agg):
    graph.vs[_comm] = [agg.vs[i][_comm] for i in graph.vs[_refineIndex]]

def renumber(graph, comm):
    mapping, nxt = {}, 0; out = []
    for c in graph.vs[comm]:
        if c not in mapping: mapping[c] = nxt; nxt += 1
        out.append(mapping[c])
    graph.vs[comm] = out

def leiden_custom(graph, attr, iterations=8, simplified=False):
    initialiseGraph(graph)
    communities = initialisePartition(graph, _comm)
    for _ in range(iterations):
        localMove(graph, communities)
        communities = cleanCommunities(communities)
        refine_communities = initialisePartition(graph, _refine)
        converged = refine(graph, communities, refine_communities, simplified)
        refine_communities = cleanCommunities(refine_communities)
        graphs = [graph]
        while not converged:
            graph = aggregate(graph, refine_communities); graphs.append(graph)
            localMove(graph, communities); communities = cleanCommunities(communities)
            converged = refine(graph, communities, refine_communities, simplified)
            refine_communities = cleanCommunities(refine_communities)
        for g_fine, g_coarse in zip(graphs[-2::-1], graphs[:0:-1]):
            deAggregate(g_fine, g_coarse)
        graph = graphs[0]
    graph.vs[attr] = graph.vs[_comm]
    for k in (_comm, _refine, _refineIndex, _queued, _degree, _selfEdges):
        if k in graph.vs.attributes(): del graph.vs[k]
    if _wellconnected in graph.vs.attributes(): del graph.vs[_wellconnected]
    if _m in graph.attributes(): del graph[_m]
    renumber(graph, attr)
    return quality(graph, attr)

# Run custom Leiden on a fresh copy of the same graph
g_custom = g.copy()
random.seed(42)  # lock randomness for localMove/refine
mod_custom = leiden_custom(g_custom, "comm", iterations=8, simplified=False)
labels_custom = np.array(g_custom.vs["comm"])
adata.obs["leiden_custom"] = pd.Categorical(labels_custom.astype(str))
print(f"Custom Leiden clusters: {adata.obs['leiden_custom'].nunique()}  |  Modularity: {mod_custom:.4f}")

def leiden_with_snapshots_projected(graph, attr="comm", iterations=6, simplified=False):
    """
    Same as leiden_with_snapshots but returns ALL frames projected to the
    ORIGINAL vertex set (length N0) so every frame has shape (N0,).
    """
    N0 = graph.vcount()
    snaps = []

    # Map from original vertex -> current graph index
    current_map = np.arange(N0, dtype=int)

    def _record_current_labels(label_attr):
        """Project labels currently on `graph` onto original vertices via current_map."""
        if label_attr in graph.vs.attributes():
            lab = np.asarray(graph.vs[label_attr], dtype=int)
            proj = lab[current_map]
        else:
            proj = np.zeros(N0, dtype=int)
        snaps.append(proj.tolist())

    # === Standard Leiden, but keep `current_map` updated across aggregations ===
    initialiseGraph(graph)
    communities = initialisePartition(graph, _comm)

    # initial labels (each node = its own community)
    _record_current_labels(_comm)

    for _ in range(iterations):
        # 1) local move on current graph
        localMove(graph, communities)
        _record_current_labels(_comm)

        communities = cleanCommunities(communities)

        # 2) refine step
        refine_communities = initialisePartition(graph, _refine)
        converged = refine(graph, communities, refine_communities, simplified)
        _record_current_labels(_refine)
        refine_communities = cleanCommunities(refine_communities)

        graphs = [graph]
        while not converged:
            # 3) aggregate (coarsen) — update current_map using the fine->coarse indices
            agg = aggregate(graph, refine_communities)  # sets graph.vs[_refineIndex] on the FINE graph
            fine_to_coarse = np.asarray(graph.vs[_refineIndex], dtype=int)
            current_map = fine_to_coarse[current_map]
            graph = agg
            graphs.append(graph)

            # 4) local move again on the coarse graph
            localMove(graph, communities)
            _record_current_labels(_comm)
            communities = cleanCommunities(communities)

            # 5) refine again
            converged = refine(graph, communities, refine_communities, simplified)
            _record_current_labels(_refine)
            refine_communities = cleanCommunities(refine_communities)

        # deaggregate labels back to the finest graph (internal bookkeeping only)
        for g_fine, g_coarse in zip(graphs[-2::-1], graphs[:0:-1]):
            deAggregate(g_fine, g_coarse)
        graph = graphs[0]

    # final labels on the finest graph
    graph.vs[attr] = graph.vs[_comm]

    # clean attributes (like your original)
    for k in (_comm, _refine, _refineIndex, _queued, _degree, _selfEdges):
        if k in graph.vs.attributes(): del graph.vs[k]
    if _wellconnected in graph.vs.attributes(): del graph.vs[_wellconnected]
    if _m in graph.attributes(): del graph[_m]
    renumber(graph, attr)

    # record the final projected labels too
    _record_current_labels(attr)

    return quality(graph, attr), snaps


# =========================================================
# 5) Compare library vs custom (counts, ARI) + UMAPs
# =========================================================
y_lib = adata.obs["leiden_lib"].astype(str).to_numpy()
y_cus = adata.obs["leiden_custom"].astype(str).to_numpy()
print("Library clusters:", len(set(y_lib)), " | Custom clusters:", len(set(y_cus)))
print("ARI (library vs custom):", ARI(y_lib, y_cus))

sc.pl.umap(adata, color=["leiden_lib", "leiden_custom"], legend_loc="on data", frameon=False, size=20)

# =========================================================
# 6) (Optional) Animation of custom color changes on UMAP
#     (re-runs custom Leiden with snapshots)
# =========================================================
def _record_snapshot(graph, attr, snaps):
    snaps.append(list(graph.vs[attr]) if attr in graph.vs.attributes() else [0]*graph.vcount())

def leiden_with_snapshots(graph, attr="comm", iterations=6, simplified=False):
    snaps = []
    initialiseGraph(graph)
    communities = initialisePartition(graph, _comm)
    _record_snapshot(graph, _comm, snaps)
    for _ in range(iterations):
        localMove(graph, communities); _record_snapshot(graph, _comm, snaps)
        communities = cleanCommunities(communities)
        refine_communities = initialisePartition(graph, _refine)
        converged = refine(graph, communities, refine_communities, simplified)
        _record_snapshot(graph, _refine, snaps)
        refine_communities = cleanCommunities(refine_communities)
        graphs = [graph]
        while not converged:
            graph = aggregate(graph, refine_communities); graphs.append(graph)
            localMove(graph, communities); _record_snapshot(graph, _comm, snaps)
            communities = cleanCommunities(communities)
            converged = refine(graph, communities, refine_communities, simplified)
            _record_snapshot(graph, _refine, snaps)
            refine_communities = cleanCommunities(refine_communities)
        for g_fine, g_coarse in zip(graphs[-2::-1], graphs[:0:-1]):
            deAggregate(g_fine, g_coarse)
        graph = graphs[0]
    graph.vs[attr] = graph.vs[_comm]
    for k in (_comm, _refine, _refineIndex, _queued, _degree, _selfEdges):
        if k in graph.vs.attributes(): del graph.vs[k]
    if _wellconnected in graph.vs.attributes(): del graph.vs[_wellconnected]
    if _m in graph.attributes(): del graph[_m]
    renumber(graph, attr)
    _record_snapshot(graph, attr, snaps)
    return quality(graph, attr), snaps

# Build animation frames
g_anim = g.copy()
random.seed(42)
_, snapshots = leiden_with_snapshots_projected(g_anim, attr="comm", iterations=6, simplified=False)
adata3 = sc.tl.umap(adata, n_components=3, copy=True)  # if not already run
X3 = adata3.obsm["X_umap"]
x3, y3, z3 = X3[:,0], X3[:,1], X3[:,2]


# === Added prints you requested previously ===
final_labels = np.asarray(snapshots[-1], dtype=int)
first_final_idx = next((i for i, fr in enumerate(snapshots)
                        if np.array_equal(np.asarray(fr, dtype=int), final_labels)),
                       len(snapshots) - 1)
changes = sum(1 for i in range(1, len(snapshots))
              if not np.array_equal(np.asarray(snapshots[i-1], dtype=int),
                                    np.asarray(snapshots[i], dtype=int)))
print(f"[Leiden snapshots] Frames before final clustering appears: {first_final_idx} / {len(snapshots)-1}")
print(f"[Leiden snapshots] Number of label-change frames: {changes}")

# ===================== PER-NODE CHANGE ANALYSIS & ALL-FRAMES DISPLAY (NEW) =====================

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import colors
_np_alias = np  # optional: alias if you want to reference np via _np_alias

def snapshots_to_matrix(snapshots):
    F = len(snapshots)
    N = len(snapshots[-1])
    M = np.zeros((F, N), dtype=int)
    for i, fr in enumerate(snapshots):
        fr = np.asarray(fr, dtype=int)
        if fr.shape[0] != N:
            raise ValueError(f"Snapshot {i} has {fr.shape[0]} nodes; expected {N}.")
        M[i] = fr
    return M

def node_change_report(snapshots, names=None):
    M = snapshots_to_matrix(snapshots)         # (F, N)
    F, N = M.shape
    last = M[-1]
    chg = (M[1:] != M[:-1])                    # shape (F-1, N)
    total_changes = chg.sum(axis=0)
    change_frames = [np.where(chg[:, j])[0] + 1 for j in range(N)]
    first_change = [int(fr[0]) if fr.size else -1 for fr in change_frames]
    last_change  = [int(fr[-1]) if fr.size else -1 for fr in change_frames]

    stabilized = []
    for j in range(N):
        k = 0
        while k < F and not np.all(M[k:, j] == last[j]):
            k += 1
        stabilized.append(k if k < F else F-1)

    df = pd.DataFrame({
        "node_index": np.arange(N, dtype=int),
        "name": list(names) if names is not None else np.arange(N, dtype=int),
        "final_label": last,
        "total_changes": total_changes,
        "first_change_frame": first_change,
        "last_change_frame": last_change,
        "stabilized_at_frame": stabilized,
    })

    def path_str(col):
        frames_to_keep = [0] + list(np.where(np.diff(col)!=0)[0] + 1)
        return "; ".join(f"{f}:{int(col[f])}" for f in frames_to_keep)

    df["label_path_compact"] = [path_str(M[:, j]) for j in range(N)]
    return df, M

def consistent_colors(all_labels, labels_now):
    idx_map = {lab: i for i, lab in enumerate(sorted(all_labels))}
    idxs = np.array([idx_map[z] for z in labels_now], dtype=int)
    cmap = plt.colormaps['tab20']
    norm = colors.Normalize(vmin=0, vmax=max(1, len(all_labels)-1))
    return cmap(norm(idxs))

def plot_leiden_grid(
    snapshots, XY, ncols=6, mark_changes=True,
    point_size=5, changed_size=18, unchanged_alpha=0.35,
    dpi=170, suptitle="Leiden progression (all frames)"):
    M = snapshots_to_matrix(snapshots)  # (F, N)
    F, N = M.shape
    x, y = XY[:, 0], XY[:, 1]
    all_labels = set(M.flatten().tolist())

    nrows = int(np.ceil(F / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.1), dpi=dpi)
    axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

    for f in range(F):
        ax = axes[f]
        cols = consistent_colors(all_labels, M[f])
        if mark_changes and f > 0:
            changed = (M[f] != M[f-1])
            ax.scatter(x[~changed], y[~changed], s=point_size, c=cols[~changed], linewidths=0, alpha=unchanged_alpha)
            ax.scatter(x[changed], y[changed], s=changed_size, c=cols[changed], linewidths=0, alpha=1.0)
        else:
            ax.scatter(x, y, s=point_size, c=cols, linewidths=0, alpha=1.0)
        ax.set_title(f"Frame {f+1}/{F}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

    for k in range(F, len(axes)):
        axes[k].axis("off")

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_node_timelines(snapshots, node_indices=None, max_nodes=60, sort_by="total_changes", names=None, dpi=170):
    df, M = node_change_report(snapshots, names)
    F, N = M.shape

    if node_indices is None:
        if sort_by == "stabilized_at":
            df = df.sort_values(["stabilized_at_frame", "total_changes"], ascending=[True, False])
        else:
            df = df.sort_values(["total_changes", "stabilized_at_frame"], ascending=[False, True])
        node_indices = df["node_index"].head(max_nodes).tolist()
    else:
        node_indices = list(node_indices)[:max_nodes]

    subM = M[:, node_indices].T  # (k, F)
    labs = sorted(set(subM.flatten().tolist()))
    cmap = plt.colormaps['tab20']
    lab_to_idx = {lab:i for i, lab in enumerate(labs)}
    sub_idx = np.vectorize(lab_to_idx.get)(subM)

    fig, ax = plt.subplots(figsize=(min(14, 0.2*F + 4), 0.28*len(node_indices) + 1), dpi=dpi)
    im = ax.imshow(sub_idx, aspect='auto', interpolation='nearest')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Nodes")
    ax.set_title("Node label timelines (rows = nodes, columns = frames)")
    ax.set_xticks(np.linspace(0, F-1, min(F, 10), dtype=int))
    ax.set_yticks(np.arange(len(node_indices)))
    ylabels = [str(df.loc[df["node_index"]==j, "name"].values[0]) for j in node_indices] if names is not None else [str(j) for j in node_indices]
    ax.set_yticklabels(ylabels)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Cluster (remapped index)")
    plt.tight_layout()
    plt.show()

# ===================== RUN REPORT + PLOTS (calls) =====================
report_df, M_mat = node_change_report(snapshots, names=adata.obs_names)
print(report_df.head(10))
report_df.to_csv("leiden_node_change_report.csv", index=False)
print("Saved: leiden_node_change_report.csv")

# =========================================================
# Back to your existing animation code
# =========================================================
XY = adata.obsm["X_umap"]; x, y = XY[:,0], XY[:,1]
all_labels_sorted = sorted(set(l for frame in snapshots for l in frame))

def palette_from_labels(all_labels, labels_now):
    idx_map = {lab:i for i, lab in enumerate(all_labels)}
    idxs = np.array([idx_map[z] for z in labels_now])
    cmap = plt.colormaps['tab20']
    norm = colors.Normalize(vmin=0, vmax=max(1, len(all_labels)-1))
    return cmap(norm(idxs))

# Grid of ALL frames with changed nodes emphasized
plot_leiden_grid(
    snapshots,
    XY=adata.obsm["X_umap"],
    ncols=6,
    mark_changes=True,
    point_size=5,
    changed_size=16,
    unchanged_alpha=0.35,
    dpi=170,
    suptitle="Leiden progression (all frames — changed nodes emphasized)"
)

# Optional: timelines of most-volatile nodes
plot_node_timelines(
    snapshots,
    node_indices=None,       # or a list like [12, 87, 104]
    max_nodes=60,
    sort_by="total_changes", # or "stabilized_at"
    names=adata.obs_names,
    dpi=170
)

# === Your existing single-animation preview (kept intact) ===
fig, ax = plt.subplots(figsize=(6.6, 6.2))
cols0 = palette_from_labels(all_labels_sorted, snapshots[0])
scat = ax.scatter(x, y, s=6, c=cols0, linewidths=0)
ax.set_title(f"Leiden progression — frame 1/{len(snapshots)}")
ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

def update(i):
    scat.set_facecolors(palette_from_labels(all_labels_sorted, snapshots[i]))
    ax.set_title(f"Leiden progression — frame {i+1}/{len(snapshots)}")
    return (scat,)

anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=650, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())

# Optional: save GIF or MP4 (to avoid notebook embed limit)
# import matplotlib as mpl; mpl.rcParams['animation.embed_limit'] = 64  # MB
# anim.save("pbmc3k_leiden_custom.mp4", writer="ffmpeg", fps=6)

# =========================================================
# 7) Save outputs
# =========================================================
adata.obs[["leiden_lib","leiden_custom"]].to_csv("pbmc3k_clusters_compare.csv")
adata.write("pbmc3k_processed_compare.h5ad")
print("Saved: pbmc3k_clusters_compare.csv, pbmc3k_processed_compare.h5ad")




# ===================== 3D ROTATING GIF (colors + changed-node emphasis, no numbers) =====================
import numpy as np, matplotlib.pyplot as plt
from matplotlib import animation, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ensure a 3D UMAP exists without touching your 2D embedding
adata3 = sc.tl.umap(adata, n_components=3, copy=True)
X3 = adata3.obsm["X_umap"]
x3, y3, z3 = X3[:,0], X3[:,1], X3[:,2]

F = len(snapshots)
all_labels_sorted = sorted(set(l for fr in snapshots for l in fr))

def _palette(all_labels_sorted, labels_now):
    idx_map = {lab:i for i, lab in enumerate(all_labels_sorted)}
    idxs = np.array([idx_map[z] for z in labels_now], dtype=int)
    cmap = plt.colormaps['tab20']
    norm = colors.Normalize(vmin=0, vmax=max(1, len(all_labels_sorted)-1))
    return cmap(norm(idxs))  # (N,4)

fig = plt.figure(figsize=(7.2, 6.4), dpi=130)
ax = fig.add_subplot(111, projection='3d')

labs0 = np.asarray(snapshots[0], dtype=int)
cols0 = _palette(all_labels_sorted, labs0)

# base scatter: all points (we set per-point RGBA every frame)
scat_base = ax.scatter(x3, y3, z3, s=8, c=cols0, depthshade=False)

# overlay for changed nodes (only those, bigger points)
scat_changed = ax.scatter([], [], [], s=28, c=np.empty((0,4)), depthshade=False)

# clean axes
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_title(f"Leiden progression (3D) — frame 1/{F}")
fig.tight_layout()

# rotation config
start_azim = 30.0
start_elev = 18.0
turns = 1.0                       # full 360° turns across all frames
azim_step = 360.0 * turns / max(1, F-1)
elev_amp = 4.0                    # gentle nod
elev_phase = np.linspace(0, 2*np.pi*turns, F)

def _update(i):
    labs = np.asarray(snapshots[i], dtype=int)
    cols = _palette(all_labels_sorted, labs)

    if i == 0:
        changed_mask = np.zeros_like(labs, dtype=bool)
    else:
        prev = np.asarray(snapshots[i-1], dtype=int)
        changed_mask = (labs != prev)

    # base layer with per-point alpha (fade changed nodes)
    rgba = cols.copy()
    rgba[:, 3] = np.where(changed_mask, 0.35, 1.0)
    scat_base._offsets3d = (x3, y3, z3)
    scat_base.set_facecolors(rgba)

    # overlay only changed nodes
    if changed_mask.any():
        scat_changed._offsets3d = (x3[changed_mask], y3[changed_mask], z3[changed_mask])
        scat_changed.set_facecolors(cols[changed_mask])
    else:
        scat_changed._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        scat_changed.set_facecolors(np.empty((0,4)))

    # rotate view
    az = start_azim + azim_step * i
    el = start_elev + elev_amp * np.sin(elev_phase[i])
    ax.view_init(elev=el, azim=az)

    ax.set_title(f"Leiden progression (3D) — frame {i+1}/{F}")
    return scat_base, scat_changed

anim3d = animation.FuncAnimation(fig, _update, frames=F, interval=450, blit=False)

# save as a single GIF (avoids notebook embed limit)
gif_path = "pbmc3k_leiden_3d_rotate.gif"
anim3d.save(gif_path, writer=animation.PillowWriter(fps=4))
plt.close(fig)
print(f"Saved 3D rotating GIF: {gif_path}")
# ===============================================================================================



# ===================== MAKE A SINGLE GIF FROM ALL FRAMES =====================
from matplotlib import animation, colors
import numpy as np, matplotlib.pyplot as plt

# --- palette helper (consistent across frames) ---
def _palette(all_labels_sorted, labels_now):
    idx_map = {lab:i for i, lab in enumerate(all_labels_sorted)}
    idxs = np.array([idx_map[z] for z in labels_now], dtype=int)
    cmap = plt.colormaps['tab20']
    norm = colors.Normalize(vmin=0, vmax=max(1, len(all_labels_sorted)-1))
    return cmap(norm(idxs))

# --- precompute ---
F = len(snapshots)
all_labels_sorted = sorted(set(l for fr in snapshots for l in fr))
sx = np.asarray(x); sy = np.asarray(y)

# --- build figure and artists once ---
fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=140)
cols0 = _palette(all_labels_sorted, snapshots[0])
# base scatter (unchanged nodes each frame will be partially faded)
scat_base = ax.scatter(sx, sy, s=6, c=cols0, linewidths=0, alpha=1.0)
# overlay for changed nodes (larger points each frame)
scat_changed = ax.scatter([], [], s=18, c=[], linewidths=0, alpha=1.0)

ax.set_title(f"Leiden progression — frame 1/{F}")
ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
fig.tight_layout()

# --- update per frame ---
def _update(i):
    labs = np.asarray(snapshots[i], dtype=int)
    cols = _palette(all_labels_sorted, labs)

    if i == 0:
        changed_mask = np.zeros_like(labs, dtype=bool)
    else:
        prev = np.asarray(snapshots[i-1], dtype=int)
        changed_mask = (labs != prev)

    # base: draw all points; fade the ones that changed (they'll be overdrawn by 'changed' overlay)
    alpha_base = np.where(changed_mask, 0.35, 1.0)  # fade changed ones underneath
    scat_base.set_offsets(np.c_[sx, sy])
    scat_base.set_facecolors(cols)
    # unfortunately matplotlib scatter can't vary alpha per point directly via set_alpha,
    # so we apply alpha into RGBA array:
    rgba = scat_base.get_facecolors()
    rgba[:, 3] = alpha_base
    scat_base.set_facecolors(rgba)

    # overlay: only changed nodes, larger markers, full opacity
    scat_changed.set_offsets(np.c_[sx[changed_mask], sy[changed_mask]])
    scat_changed.set_facecolors(cols[changed_mask] if changed_mask.any() else [])

    ax.set_title(f"Leiden progression — frame {i+1}/{F}")
    return scat_base, scat_changed

anim = animation.FuncAnimation(fig, _update, frames=F, interval=450, blit=False)

# --- save as GIF (no notebook embed limit) ---
gif_path = "pbmc3k_leiden_all_frames.gif"
anim.save(gif_path, writer=animation.PillowWriter(fps=4))
plt.close(fig)
print(f"Saved GIF: {gif_path}")
# ============================================================================


