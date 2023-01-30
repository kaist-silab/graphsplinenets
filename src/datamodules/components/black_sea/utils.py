from dataclasses import dataclass, replace
from typing import Generic, TypeVar, Union

import torch
from torchtyping import TensorType

from functools import cached_property
from itertools import combinations
from typing import Optional

import numpy as np
import skfem
import torch
from scipy.spatial import ConvexHull, Delaunay


def inside_angle(a, b, c):
    """Compute the inside angle of a right triangle at a with a right angle at b."""

    opposite = np.linalg.norm(b - c)
    hypotenuse = np.linalg.norm(a - c)

    # Due to floating point errors, this ratio can actually become larger than 1, so we
    # cap it at 1 to avoid a warning
    ratio = min(opposite / hypotenuse, 1.0)
    return np.arcsin(ratio)


def project_onto(p, line_p):
    assert len(line_p) == 2, "Can only project onto lines"
    a, b = line_p
    ab = b - a
    ap = p - a
    ab_norm = ab / np.linalg.norm(ab)
    return a + (np.inner(ap, ab_norm)) * ab_norm


class CellPredicate:
    def __call__(
        self, cell_idx: int, cell: list[int], boundary_faces: list[list[int]]
    ) -> bool:
        """Decide if a cell fulfills the predicate.

        Arguments
        ---------
        cell_idx
            Index of the cell in tri.simplices
        cell
            Node indices of the cell vertices
        boundary_faces
            Faces of the cell that are on the boundary of the mesh
        """

        raise NotImplementedError()


def select_boundary_mesh_cells(tri: Delaunay, predicate: CellPredicate) -> np.ndarray:
    """Delete mesh cells from the boundary of a mesh that fulfill a predicate.

    This function implements the iterative boundary filtering algorithm described in
    Appendix F.

    Returns
    -------
    A mask over `tri.points` that selects nodes to delete
    """

    # Put the vertex indices of a boundary edge in a canonical order
    e = lambda nodes: tuple(sorted(nodes))

    cells = tri.simplices
    n_cells = len(cells)
    n_vertices = cells.shape[-1]
    n_face_vertices = n_vertices - 1
    adjacent = [[] for i in range(n_cells)]
    node_sets = [set(cells[i]) for i in range(n_cells)]
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            if len(node_sets[i] & node_sets[j]) == n_face_vertices:
                adjacent[i].append(j)
                adjacent[j].append(i)

    delete_cells = np.zeros(len(cells), dtype=bool)
    visited = np.zeros(len(cells), dtype=bool)
    boundary_faces = set(e(nodes) for nodes in tri.convex_hull)
    faces_on_boundary = np.array(
        [
            sum(
                [
                    e(nodes) in boundary_faces
                    for nodes in combinations(cell, n_face_vertices)
                ]
            )
            for cell in cells
        ]
    )
    boundary_stack = [
        (i, cell) for i, cell in enumerate(cells) if faces_on_boundary[i] > 0
    ]
    while len(boundary_stack) > 0:
        i, cell = boundary_stack.pop()
        visited[i] = True

        # Separate boundary and interior nodes
        cell_boundary_faces = [
            nodes
            for nodes in combinations(cell, n_face_vertices)
            if e(nodes) in boundary_faces
        ]

        if predicate(i, cell, cell_boundary_faces):
            delete_cells[i] = True

            # If a mesh cell is filtered, mark all its faces as boundary (even the
            # exterior ones; they don't matter and it simplifies the code)
            for nodes in combinations(cell, n_face_vertices):
                boundary_faces.add(e(nodes))

            # The adjacent interior cells have now become boundary cells and need
            # to be inspected as well
            for j in adjacent[i]:
                if delete_cells[j] or visited[j]:
                    continue

                boundary_stack.append((j, cells[j]))

    return delete_cells


@dataclass
class BoundaryAnglePredicate(CellPredicate):
    """Check if a boundary cell is too acute/elongated.

    A cell is flagged as too acute if its interior node is too close to the boundary.
    """

    p: np.ndarray
    epsilon: float = np.pi / 180

    def __call__(self, cell_idx, cell, boundary_faces):
        if len(boundary_faces) != 1:
            # Keep any cell that is on an edge or corner of the domain
            return False

        boundary_nodes = boundary_faces[0]
        interior_node = list(set(cell) - set(boundary_nodes))[0]

        # Project the interior node onto the boundary surface
        interior_p = self.p[interior_node]
        interior_projection = project_onto(interior_p, self.p[list(boundary_nodes)])

        # Compute the inside angle of the right triangle (interior_p, a,
        # interior_projection) for all boundary nodes a
        angles = [
            inside_angle(self.p[a], interior_projection, interior_p)
            for a in boundary_nodes
        ]

        # Minimum inside angle at the boundary points
        min_angle = min(angles)

        if np.isnan(min_angle):
            # I have observed this for some degenerate triangles, so we just filter them
            # out
            return True
        else:
            return min_angle < self.epsilon


def select_acute_boundary_triangles(
    tri: Delaunay, epsilon: float = np.pi / 180
) -> np.ndarray:
    """Select highly acute triangle artifacts from the boundary of a Delaunay
    triangulation.

    It is important to note that multiple (almost, up to float precision) linear points
    (i.e. on a line), can create layers of these acute triangles. This necessitates the
    iterative algorithm below.

    Parameters
    ----------
    tri
        Delaunay triangulation
    epsilon
        Maximum angle to filter in radians
    """

    predicate = BoundaryAnglePredicate(tri.points, epsilon)
    return select_boundary_mesh_cells(tri, predicate)


@dataclass
class Domain:
    """A fixed observation domain consisting of nodes and a mesh.

    Attributes
    ----------
    x : ndarray of size n_nodes x space_dim
        Location of the nodes
    mesh
        A mesh of the nodes. You can optionally pass in a fixed or precomputed mesh.
    fixed_values_mask : boolean ndarray of size n_nodes
        An optional mask that selects nodes where the values are fixed and therefore no
        prediction should be made
    """

    x: np.ndarray
    # You can pass None but after __post_init__ this attribute will always hold a mesh, so
    # it is not marked as Optional
    mesh: skfem.Mesh = None
    fixed_values_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape[1] in (1, 2, 3)

        if self.mesh is None:
            self.mesh = self._generate_mesh()

        self._reorder_vertices()

    def normalize(self) -> "Domain":
        """Normalize the node coordinates to mean 0 and mean length of 1."""
        mean = self.x.mean(axis=0, keepdims=True)
        std = np.linalg.norm(self.x - mean, axis=-1).mean()
        x = (self.x - mean) / std
        mesh = self.mesh
        if mesh is not None:
            mesh = replace(mesh, doflocs=x.T)
        return replace(self, x=x, mesh=mesh)

    def __str__(self):
        return f"<{len(self)} points in {self.dim}D; min={self.x.min():.3f}, max={self.x.max():.3f}>"

    def __len__(self):
        return self.x.shape[0]

    @property
    def dim(self):
        return self.x.shape[1]

    @cached_property
    def basis(self):
        return skfem.CellBasis(self.mesh, self.mesh.elem())

    def _generate_mesh(self):
        doflocs = np.ascontiguousarray(self.x.T)
        if self.dim == 1:
            return skfem.MeshLine(np.sort(doflocs))
        elif self.dim == 2:
            tri = Delaunay(self.x)
            simplices = tri.simplices[~select_acute_boundary_triangles(tri)]
            return skfem.MeshTri(doflocs, np.ascontiguousarray(simplices.T))
        elif self.dim == 3:
            tri = Delaunay(self.x)
            simplices = tri.simplices[~select_acute_boundary_triangles(tri)]
            return skfem.MeshTet(doflocs, np.ascontiguousarray(simplices.T))

    def _reorder_vertices(self):
        # Put cell vertices into some "canonical" order to make it easier for models to
        # generalize
        if self.dim in (2, 3):
            cells = self.mesh.t.T
            vertices = self.x[cells]
            cell_centers = vertices.mean(axis=1)
            # Convert into cell-local coordinates
            vertex_local = vertices - cell_centers[:, None, :]

            # Compute the angle of the vertices' polar coordinates. We do the same thing in
            # both 2 and 3 dimensions which corresponds to projecting 3D points onto the 2D
            # plane first. Is there something better?
            theta = np.arctan2(vertex_local[..., 1], vertex_local[..., 0])

            order = np.argsort(theta, axis=1)
            cells = np.take_along_axis(cells, order, axis=1)

            self.mesh = replace(self.mesh, t=np.ascontiguousarray(cells.T))


class TimeEncoder:
    """A time encoder encodes strictly increasing timestamps into other time
    representations.

    This can, for example, be used to implement periodically recurring behavior in
    non-autonomous models.
    """

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        raise NotImplementedError()


@dataclass
class InvariantEncoder(TimeEncoder):
    """Make all sequences start at 0 when they are translation invariant."""

    start: TensorType["batch"]

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        return (t - self.start).unsqueeze(dim=-1)


@dataclass
class PeriodicEncoder(TimeEncoder):
    """Encode linear, periodic time into 2-dimensional positions on a circle.

    This ensures that the time features have a smooth transition at the periodicity
    boundary, e.g. new year's eve for data with yearly periodicity.
    """

    base: TensorType["batch"]
    period: TensorType["batch"]

    def encode(self, t: TensorType["batch"]) -> TensorType["batch", "time_encoding"]:
        phase = (t - self.base) * (2 * torch.pi / self.period)
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


StandardizerTensor = Union[
    TensorType["feature"],
    TensorType["node", "feature"],
    TensorType["time", "node", "feature"],
    TensorType["batch", "time", "node", "feature"],
]
DomainInfo = TypeVar("DomainInfo")


@dataclass
class Standardizer:
    """Node feature standardization

    The standardization means and standard deviation can optionally be node-, time-, and
    batch-specific.
    """

    mean: StandardizerTensor
    std: StandardizerTensor

    def __post_init__(self):
        assert self.mean.shape == self.std.shape
        assert 1 <= self.mean.ndim <= 4

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def slice_time(self, start: int, end: int):
        if self.mean.ndim == 3:
            return Standardizer(self.mean[start:end], self.std[start:end])
        elif self.mean.ndim == 4:
            return Standardizer(self.mean[:, start:end], self.std[:, start:end])
        else:
            return self


@dataclass
class STBatch(Generic[DomainInfo]):
    """A batch of spatio-temporal data.

    This type is the common interface for the provided and custom data loaders.

    Attributes
    ----------
    domain
        The domain that the data in this batch is from.
    domain_info
        Precomputed information domain information that the model needs for training as
        tensors so that it can be transferred to the GPU.
    t
        Batched time steps
    u
        Batched sensor data
    context_steps
        Says how many time steps in the beginning of the batch are meant as context for
        the prediction. The rest is meant as prediction targets.
    standardizer
        Node feature standardizer
    time_encoder
        Encodes the time steps in `t` into the values that are actually fed to
        non-autonomous models.
    """

    domain: Domain
    domain_info: DomainInfo
    t: TensorType["batch", "time"]
    u: TensorType["batch", "time", "node", "feature"]
    context_steps: int
    standardizer: Standardizer
    time_encoder: TimeEncoder

    @property
    def batch_size(self) -> int:
        return self.u.shape[0]

    @property
    def context_t(self) -> TensorType["batch", "context_time"]:
        return self.t[:, : self.context_steps]

    @property
    def target_t(self) -> TensorType["batch", "target_time"]:
        return self.t[:, self.context_steps :]

    @property
    def context_u(self) -> TensorType["batch", "context_time", "node", "feature"]:
        return self.u[:, : self.context_steps]

    @property
    def target_u(self) -> TensorType["batch", "target_time", "node", "feature"]:
        return self.u[:, self.context_steps :]

    @property
    def context_standardizer(self):
        return self.standardizer.slice_time(0, self.context_steps)

    @property
    def target_standardizer(self):
        return self.standardizer.slice_time(self.context_steps, None)


@dataclass(frozen=True)
class MeshConfig:
    """Configuration for the generation of a sparse mesh from a larger set of points.

    Attributes
    ----------
    k
        Number of nodes to select
    epsilon
        Maximum angle of boundary cells to filter out in degrees
    seed
        Random seed for reproducibility
    """

    k: int
    epsilon: float
    seed: int

    def random_state(self):
        return np.random.RandomState(int(self.seed) % 2 ** 32)

    def epsilon_radians(self):
        return self.epsilon * np.pi / 180

    def angle_predicate(self, tri: Delaunay):
        return BoundaryAnglePredicate(tri.points, self.epsilon_radians())


def sample_mesh(
    config: MeshConfig,
    points: np.ndarray,
    predicate_factory: Optional[Callable[[Delaunay], CellPredicate]] = None,
) -> tuple[np.ndarray, Domain]:
    """Create a domain from a subset of points, optionally filtering out some cells.

    Returns
    -------
    Indices of the points that were selected as mesh nodes and the domain
    """

    import skfem
    from sklearn_extra.cluster import KMedoids

    # Select k sparse observation points uniformly-ish
    km = KMedoids(
        n_clusters=config.k, init="k-medoids++", random_state=config.random_state()
    )
    km.fit(points)
    node_indices = km.medoid_indices_

    # Mesh the points with Delaunay triangulation
    tri = Delaunay(points[node_indices])

    # Filter out mesh boundary cells that are too acute or contain mostly land
    if predicate_factory is not None:
        predicate = predicate_factory(tri)
        filter = select_boundary_mesh_cells(tri, predicate)
        tri.simplices = tri.simplices[~filter]

    # Ensure that every node is in at least one mesh cell
    cell_counts = np.zeros(config.k, dtype=int)
    np.add.at(cell_counts, tri.simplices, 1)
    assert all(cell_counts >= 1)

    mesh = skfem.MeshTri(
        np.ascontiguousarray(tri.points.T), np.ascontiguousarray(tri.simplices.T)
    )
    domain = Domain(tri.points, mesh=mesh)
    return node_indices, domain
