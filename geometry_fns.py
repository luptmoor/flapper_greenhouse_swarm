import numpy as np
from scipy.spatial import ConvexHull
import unittest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def separating_axis_theorem(poly1, poly2):
    """
    Checks intersection between two convex polyhedra using the Separating Axis Theorem.
    poly1, poly2: np.ndarray of shape (N, 3) and (M, 3)
    Returns True if they intersect, False otherwise.
    """
    def get_faces(points):
        hull = ConvexHull(points)
        return hull.simplices

    def get_normals(points, faces):
        normals = []
        for face in faces:
            v0, v1, v2 = points[face[0]], points[face[1]], points[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm
                normals.append(normal)
        return normals

    # Get faces and normals for both polyhedra
    faces1 = get_faces(poly1)
    faces2 = get_faces(poly2)
    normals1 = get_normals(poly1, faces1)
    normals2 = get_normals(poly2, faces2)

    # Get edge directions
    def get_edges(points, faces):
        edges = set()
        for face in faces:
            for i in range(3):
                a, b = sorted((face[i], face[(i+1)%3]))
                edges.add((a, b))
        return [points[b] - points[a] for a, b in edges]

    edges1 = get_edges(poly1, faces1)
    edges2 = get_edges(poly2, faces2)

    # Axes to test: face normals and cross products of edges
    axes = normals1 + normals2
    for e1 in edges1:
        for e2 in edges2:
            axis = np.cross(e1, e2)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                axes.append(axis / norm)

    # Project both polyhedra onto each axis and check for separation
    for axis in axes:
        proj1 = np.dot(poly1, axis)
        proj2 = np.dot(poly2, axis)
        if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
            return False  # Found a separating axis

    return True  # No separating axis found

def pyramid_cuboid_intersect(pyramid_points, cuboid_points):
    """
    pyramid_points: np.ndarray of shape (5, 3)
    cuboid_points: np.ndarray of shape (8, 3)
    Returns True if the pyramid and cuboid intersect, False otherwise.
    """
    pyramid_points = np.asarray(pyramid_points)
    cuboid_points = np.asarray(cuboid_points)
    if pyramid_points.shape != (5, 3) or cuboid_points.shape != (8, 3):
        raise ValueError("pyramid_points must be (5,3), cuboid_points must be (8,3)")
    return separating_axis_theorem(pyramid_points, cuboid_points)



import matplotlib.pyplot as plt

def random_cuboid(center, size):
    """Generate 8 points of a cuboid given center and size (length, width, height)."""
    l, w, h = size
    cx, cy, cz = center
    # 8 corners
    corners = np.array([
        [cx - l/2, cy - w/2, cz - h/2],
        [cx + l/2, cy - w/2, cz - h/2],
        [cx + l/2, cy + w/2, cz - h/2],
        [cx - l/2, cy + w/2, cz - h/2],
        [cx - l/2, cy - w/2, cz + h/2],
        [cx + l/2, cy - w/2, cz + h/2],
        [cx + l/2, cy + w/2, cz + h/2],
        [cx - l/2, cy + w/2, cz + h/2],
    ])
    return corners

def random_pyramid(base_center, base_size, height):
    """Generate 5 points of a pyramid (square base, apex above center)."""
    l, w = base_size
    cx, cy, cz = base_center
    # 4 base corners
    base = np.array([
        [cx - l/2, cy - w/2, cz],
        [cx + l/2, cy - w/2, cz],
        [cx + l/2, cy + w/2, cz],
        [cx - l/2, cy + w/2, cz],
    ])
    apex = np.array([[cx, cy, cz + height]])
    return np.vstack([base, apex])

def plot_poly(ax, points, color, alpha=0.5):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        tri = Poly3DCollection([points[simplex]], color=color, alpha=alpha)
        ax.add_collection3d(tri)


if __name__ == "__main__":
    while True:
        # Generate random cuboid
        np.random.seed()  # Use system time for randomness in each iteration
        domain_min, domain_max = 0, 10

        # Cuboid
        cuboid_center = np.random.uniform(domain_min+2, domain_max-2, size=3)
        cuboid_size = np.random.uniform(1, 3, size=3)
        cuboid_pts = random_cuboid(cuboid_center, cuboid_size)

        # Pyramid (pointing horizontally, rotated about z)
        pyramid_base_center = np.random.uniform(domain_min+2, domain_max-2, size=3)
        pyramid_base_center[2] = np.random.uniform(domain_min+1, domain_max-4)  # keep base lower
        pyramid_base_size = np.random.uniform(1, 3, size=2)
        pyramid_length = np.random.uniform(1, 3)  # "height" along x axis

        # Create pyramid with base in yz-plane, apex along +x
        l, w = pyramid_base_size
        cx, cy, cz = pyramid_base_center
        base = np.array([
            [cx, cy - l/2, cz - w/2],
            [cx, cy + l/2, cz - w/2],
            [cx, cy + l/2, cz + w/2],
            [cx, cy - l/2, cz + w/2],
        ])
        apex = np.array([[cx + pyramid_length, cy, cz]])
        pyramid_pts = np.vstack([base, apex])

        # Rotate pyramid about z axis
        theta = np.random.uniform(0, 2*np.pi)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        pyramid_pts = (Rz @ (pyramid_pts - pyramid_base_center).T).T + pyramid_base_center

        # Check intersection
        intersect = pyramid_cuboid_intersect(pyramid_pts, cuboid_pts)
        print("Do the pyramid and cuboid intersect?", intersect)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_poly(ax, cuboid_pts, color='blue', alpha=0.4)
        plot_poly(ax, pyramid_pts, color='green', alpha=0.6)
        ax.scatter(*cuboid_pts.T, color='blue')
        ax.scatter(*pyramid_pts.T, color='green')
        ax.set_xlim(domain_min, domain_max)
        ax.set_ylim(domain_min, domain_max)
        ax.set_zlim(domain_min, domain_max)
        plt.title(f"Intersection: {intersect}")
        plt.show()
