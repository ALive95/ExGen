import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box as sbox

np.random.seed(7)

# ── Grid & function (R²→R²) ──────────────────────────────────────────────────
N = 200
x1d = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x1d, x1d)
Z = (1 + np.sin(np.pi * X) * np.cos(np.pi * Y)) / 2
W = np.cos(np.pi * X) * np.sin(np.pi * Y)

cmap_name = 'inferno'
norm = mcolors.Normalize(W.min(), W.max())
cmap = plt.colormaps[cmap_name]

# ── Sample points ────────────────────────────────────────────────────────────
n_pts = 200
pts = np.random.uniform(-0.92, 0.92, (n_pts, 2))
z_true = (1 + np.sin(np.pi * pts[:, 0]) * np.cos(np.pi * pts[:, 1])) / 2
w_true = np.cos(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])

# ── Bounded Voronoi (mirror points) ─────────────────────────────────────────
def bounded_voronoi_cells(pts):
    bbox = sbox(-1, -1, 1, 1)
    d = 2.0
    mirrors = np.vstack([pts,
                         pts + [2*d, 0], pts - [2*d, 0],
                         pts + [0, 2*d], pts - [0, 2*d]])
    vor = Voronoi(mirrors)
    cells = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or not reg:
            cells.append(None); continue
        poly = Polygon(vor.vertices[reg]).intersection(bbox)
        if poly.is_empty or poly.geom_type != 'Polygon':
            cells.append(None)
        else:
            cells.append(np.array(poly.exterior.coords[:-1]))
    return cells

cells = bounded_voronoi_cells(pts)

# ── Histogram estimator ──────────────────────────────────────────────────────
n_cells = 10
edges = np.linspace(-1, 1, n_cells + 1)

def build_grid_estimates(pts, vals, edges):
    n = len(edges) - 1
    sums = np.full((n, n), np.nan)
    counts = np.zeros((n, n))
    val_acc = np.zeros((n, n))
    for p, v in zip(pts, vals):
        ix = min(np.searchsorted(edges[1:], p[0]), n - 1)
        iy = min(np.searchsorted(edges[1:], p[1]), n - 1)
        val_acc[iy, ix] += v
        counts[iy, ix] += 1
    mask = counts > 0
    sums[mask] = val_acc[mask] / counts[mask]
    return sums

grid_z = build_grid_estimates(pts, z_true, edges)
grid_w = build_grid_estimates(pts, w_true, edges)

# ── 3-D Voronoi bars ─────────────────────────────────────────────────────────
def draw_3d_voronoi(ax, cells, z_vals, c_vals):
    for cell, zv, cv in zip(cells, z_vals, c_vals):
        if cell is None:
            continue
        color = cmap(norm(cv))
        n = len(cell)
        top = [(x, y, zv) for x, y in cell]
        bot = [(x, y, 0)  for x, y in cell]
        faces = [top, bot]
        for j in range(n):
            j2 = (j + 1) % n
            faces.append([(cell[j][0],  cell[j][1],  0),
                          (cell[j2][0], cell[j2][1], 0),
                          (cell[j2][0], cell[j2][1], zv),
                          (cell[j][0],  cell[j][1],  zv)])
        pc = Poly3DCollection(faces, alpha=0.92, linewidth=0.25)
        pc.set_facecolor(color); pc.set_edgecolor('black')
        ax.add_collection3d(pc)
    _style_3d(ax)

# ── 3-D histogram bars ───────────────────────────────────────────────────────
def draw_3d_grid(ax, edges, z_grid, c_grid):
    n = len(edges) - 1
    for iy in range(n):
        for ix in range(n):
            zv = z_grid[iy, ix]; cv = c_grid[iy, ix]
            if np.isnan(zv):
                continue
            x0, x1 = edges[ix], edges[ix + 1]
            y0, y1 = edges[iy], edges[iy + 1]
            color = cmap(norm(cv))
            corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
            top = [(x, y, zv) for x, y in corners]
            bot = [(x, y, 0)  for x, y in corners]
            faces = [top, bot]
            for j in range(4):
                j2 = (j + 1) % 4
                faces.append([(corners[j][0],  corners[j][1],  0),
                              (corners[j2][0], corners[j2][1], 0),
                              (corners[j2][0], corners[j2][1], zv),
                              (corners[j][0],  corners[j][1],  zv)])
            pc = Poly3DCollection(faces, alpha=0.92, linewidth=0.25)
            pc.set_facecolor(color); pc.set_edgecolor('black')
            ax.add_collection3d(pc)
    _style_3d(ax)

def _style_3d(ax):
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.set_xlabel('$x_1$', labelpad=2); ax.set_ylabel('$x_2$', labelpad=2)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([])
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('lightgrey')

# ── 2-D partitions ───────────────────────────────────────────────────────────
def draw_2d_voronoi(ax, cells, c_vals, pts):
    for cell, cv in zip(cells, c_vals):
        if cell is None:
            continue
        ax.add_patch(plt.Polygon(cell, fc=cmap(norm(cv)), ec='grey', lw=0.6))
    ax.scatter(pts[:, 0], pts[:, 1], c='white', s=12, zorder=5,
               edgecolors='grey', linewidths=0.4)
    _style_2d(ax)

def draw_2d_grid(ax, edges, c_grid, pts):
    n = len(edges) - 1
    for iy in range(n):
        for ix in range(n):
            cv = c_grid[iy, ix]
            color = '#aaaaaa' if np.isnan(cv) else cmap(norm(cv))
            x0, x1 = edges[ix], edges[ix + 1]
            y0, y1 = edges[iy], edges[iy + 1]
            ax.add_patch(plt.Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
                                     fc=color, ec='grey', lw=0.6))
    ax.scatter(pts[:, 0], pts[:, 1], c='white', s=12, zorder=5,
               edgecolors='grey', linewidths=0.4)
    _style_2d(ax)

def _style_2d(ax):
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect('equal'); ax.axis('off')

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 8))
fig.patch.set_facecolor('white')
elev, azim = 28, -55

# Row 1: 3D
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
facecolors = cmap(norm(W))
ax1.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, alpha=1.0)
_style_3d(ax1)
ax1.view_init(elev=elev, azim=azim)
ax1.set_title("Target function $y$", pad=6)

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
draw_3d_grid(ax2, edges, grid_z, grid_w)
ax2.view_init(elev=elev, azim=azim)
ax2.set_title("Histogram estimator", pad=6)

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
draw_3d_voronoi(ax3, cells, z_true, w_true)
ax3.view_init(elev=elev, azim=azim)
ax3.set_title("Voronoi estimator", pad=6)

# Row 2: 2D
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(W, extent=[-0.9, 0.9, -0.9, 0.9], origin='lower',
           cmap=cmap_name, norm=norm, aspect='equal')
ax4.axis('off')
ax4.set_title("Target function $y$", pad=6)

ax5 = fig.add_subplot(2, 3, 5)
draw_2d_grid(ax5, edges, grid_w, pts)
ax5.set_title("Histogram partition", pad=6)

ax6 = fig.add_subplot(2, 3, 6)
draw_2d_voronoi(ax6, cells, w_true, pts)
ax6.set_title("Voronoi tessellation", pad=6)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
fig.colorbar(sm, cax=cbar_ax)

fig.subplots_adjust(left=0.03, right=0.9, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)
import matplotlib.patches as mpatches
empty_patch = mpatches.Patch(facecolor='#aaaaaa', edgecolor='grey',
                             linewidth=0.6, label='Empty cell')
fig.legend(handles=[empty_patch], loc='center',
           bbox_to_anchor=(0.9375, 0.07), frameon=False, fontsize=11)
plt.savefig('partition_figure.pdf', dpi=600, bbox_inches='tight', format='pdf')
plt.show()
print("saved")