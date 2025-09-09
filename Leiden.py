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
from matplotlib import animation, colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
#    (Scanpy default is CPM/RB; override to Modularity like your custom code)
# =========================================================
sc.tl.leiden(
    adata,
    key_added="leiden_lib",
    partition_type=la.ModularityVertexPartition,  # match your custom objective
    resolution=None,                               # <-- IMPORTANT: modularity has no resolution
    flavor="leidenalg",                            # keep current backend
    directed=False,                                # neighbors graph is undirected
    use_weights=True,
    n_iterations=-1,                               # run optimiser to convergence (leidenalg)
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
                update_communities(refine_communities, current_ref, target, community_edges, graph.vs[v][_multiplicity], self_edges, degree)
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

# =========================================================
# 5) Compare library vs custom (counts, ARI) + UMAPs
# =========================================================
y_lib = adata.obs["leiden_lib"].astype(str).to_numpy()
y_cus = adata.obs["leiden_custom"].astype(str).to_numpy()
print("Library clusters:", len(set(y_lib)), " | Custom clusters:", len(set(y_cus)))
print("ARI (library vs custom):", ARI(y_lib, y_cus))

sc.pl.umap(adata, color=["leiden_lib", "leiden_custom"], legend_loc="on data", frameon=False, size=20)

# =========================================================
# 6) Animation of custom color changes on UMAP
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
_, snapshots = leiden_with_snapshots(g_anim, attr="comm", iterations=6, simplified=False)
XY = adata.obsm["X_umap"]; x, y = XY[:,0], XY[:,1]
all_labels_sorted = sorted(set(l for frame in snapshots for l in frame))

def palette_from_labels(all_labels, labels_now):
    idx_map = {lab:i for i, lab in enumerate(all_labels)}
    idxs = np.array([idx_map[z] for z in labels_now])
    cmap = plt.colormaps['tab20']
    norm = colors.Normalize(vmin=0, vmax=max(1, len(all_labels)-1))
    return cmap(norm(idxs))

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

# Optional: save GIF
# anim.save("pbmc3k_leiden_custom.gif", writer=animation.PillowWriter(fps=2))
# print("Saved: pbmc3k_leiden_custom.gif")

# =========================================================
# 7) Save outputs
# =========================================================
adata.obs[["leiden_lib","leiden_custom"]].to_csv("pbmc3k_clusters_compare.csv")
adata.write("pbmc3k_processed_compare.h5ad")
print("Saved: pbmc3k_clusters_compare.csv, pbmc3k_processed_compare.h5ad")



from mpl_toolkits.mplot3d import Axes3D  # enables 3D projection

# Ensure 3D embedding exists
sc.tl.umap(adata, n_components=3)
umap3d = adata.obsm["X_umap"]

def plot_umap3d(labels, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        umap3d[:,0], umap3d[:,1], umap3d[:,2],
        c=pd.Categorical(labels).codes,
        cmap="tab20", s=10, alpha=0.8
    )
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    plt.show()

plot_umap3d(adata.obs["leiden_lib"], "3D UMAP — Leiden (library)")
plot_umap3d(adata.obs["leiden_custom"], "3D UMAP — Leiden (custom)")




# make sure 3D UMAP exists
sc.tl.umap(adata, n_components=3)
umap3d = adata.obsm["X_umap"]
x, y, z = umap3d[:,0], umap3d[:,1], umap3d[:,2]

def _stable_palette(labels):
    cats = pd.Categorical(labels)
    idxs = cats.codes
    cmap = plt.colormaps['tab20']
    base = cmap(np.linspace(0, 1, cmap.N))
    return base[idxs % cmap.N], cats

def umap3d_gif_detailed(
    labels,
    title,
    filename="umap3d_detailed.gif",
    n_frames=180,           # more frames => smoother & slower
    elev=28,
    spin_degrees=540,       # 1.5 turns so you see all sides
    dpi=220,                # crisper
    point_size=18,          # bigger points
    zoom=(1.0, 0.65),       # zoom in more during spin
    highlight="sequential", # None | "sequential"
    dim_alpha=0.03,         # dim non-highlighted clusters harder
    show_centroids=True,
    depthshade=False        # keep color consistent with distance
):
    cols_base, cats = _stable_palette(labels)
    labels = np.asarray(pd.Categorical(labels).codes)
    uniq = np.unique(labels)
    cluster_ix = {c: np.where(labels==c)[0] for c in uniq}

    # sort once by z so nearer points draw on top (less occlusion)
    order = np.argsort(z)
    xs, ys, zs = x[order], y[order], z[order]
    cols_sorted = cols_base[order]
    labs_sorted = labels[order]

    def _lims(arr, pad=0.05):
        lo, hi = np.min(arr), np.max(arr)
        rng = hi - lo
        return lo - pad*rng, hi + pad*rng
    xlim0, ylim0, zlim0 = _lims(xs), _lims(ys), _lims(zs)

    fig = plt.figure(figsize=(8,7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # outline underlayer + points (draw per cluster to reduce occlusion bias)
    under_handles, scat_handles = [], []
    for c in uniq:
        idx = np.where(labs_sorted==c)[0]
        under_handles.append(ax.scatter(xs[idx], ys[idx], zs[idx],
                                        s=point_size*2.0, c="k", alpha=0.16,
                                        depthshade=depthshade))
        scat_handles.append(ax.scatter(xs[idx], ys[idx], zs[idx],
                                       s=point_size, c=cols_sorted[idx],
                                       alpha=0.98, depthshade=depthshade))

    # centroids
    texts = []
    if show_centroids:
        for c in uniq:
            idx = cluster_ix[c]
            if idx.size == 0: continue
            cx, cy, cz = x[idx].mean(), y[idx].mean(), z[idx].mean()
            texts.append(ax.text(cx, cy, cz, str(c), fontsize=10, color="black", alpha=0.85))

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()

    def set_zoom(t):
        zmin, zmax = zoom
        f = 0.5*(1 - np.cos(2*np.pi*t))  # 0..1..0
        zf = zmin*(1-f) + zmax*f
        def interp(lim):
            mid = 0.5*(lim[0]+lim[1]); half = 0.5*(lim[1]-lim[0])*zf
            return (mid - half, mid + half)
        ax.set_xlim(*interp(xlim0))
        ax.set_ylim(*interp(ylim0))
        ax.set_zlim(*interp(zlim0))

    def update(i):
        # camera
        azim = (i / n_frames) * spin_degrees
        ax.view_init(elev=elev, azim=azim)
        set_zoom(i / n_frames)

        # highlight one cluster at a time (optional)
        if highlight == "sequential":
            c = uniq[(i * len(uniq)) // n_frames]
            for k, cval in enumerate(uniq):
                alpha = 0.98 if cval == c else dim_alpha
                scat_handles[k].set_alpha(alpha)
        else:
            for h in scat_handles: h.set_alpha(0.98)

        ax.set_title(f"{title} — frame {i+1}/{n_frames}")
        return tuple(scat_handles) + tuple(under_handles) + tuple(texts)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=60, blit=True)
    anim.save(filename, writer=animation.PillowWriter(fps=24))
    plt.close(fig)
    print(f"Saved: {filename}")

