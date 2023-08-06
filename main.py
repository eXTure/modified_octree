import logging
import laspy
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class PointCloud:
    def __init__(self, filename):
        self.file_name = filename
        self.points = None
        self.load_points()

    def load_points(self):
        logger.info("Loading LAS file...")
        in_file = laspy.read(self.file_name)
        self.points = np.vstack((in_file.x, in_file.y, in_file.z)).T


class Octree:
    def __init__(self, points, min_cube_size, max_depth=10):
        self.points = points
        self.min_cube_size = min_cube_size
        self.max_depth = max_depth
        self.cube_centers = [[] for _ in range(max_depth)]
        self.cube_radii = [[] for _ in range(max_depth)]
        self.cube_points_indices = []
        self.build_tree()

    def build_tree(self):
        logger.info("Constructing Octree...")
        self._build_tree_recursive(self.points, depth=0)

    def _build_tree_recursive(self, points, depth):
        if len(points) <= 1 or depth >= self.max_depth:
            return

        if len(points) <= 10000:  # Adjust the chunk size as needed
            # Calculate the bounding box of the current chunk
            min_x, min_y, min_z = np.min(points, axis=0)
            max_x, max_y, max_z = np.max(points, axis=0)
            center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2])
            radius = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2

            # Calculate points inside the cube
            inside_indices = np.where(np.all(np.abs(points - center) <= radius, axis=1))[0]
            inside_points = points[inside_indices]

            if len(inside_points) == 0:
                return

            # Calculate sphere for the chunk
            sphere_center, sphere_radius = self.calculate_sphere(inside_points)

            self.cube_centers[depth].append(sphere_center)
            self.cube_radii[depth].append(sphere_radius)
            self.cube_points_indices.append(inside_indices)
        else:
            # Split points into chunks and process each chunk recursively
            chunks = np.array_split(points, 10)  # Split points into 10 chunks (adjust as needed)
            for chunk in chunks:
                self._build_tree_recursive(chunk, depth + 1)

    def calculate_sphere(self, cube_points):
        center = np.mean(cube_points, axis=0)
        radius = max(np.linalg.norm(cube_points - center, axis=1))
        return center, radius

    def visualize(self):
        logger.info("Visualizing Octree...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.points[:, 0], self.points[:, 1], self.points[:, 2], s=1, c=self.points[:, 2], cmap="viridis", alpha=0.1
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.show()


if __name__ == "__main__":
    las_file = "2743_1234.las"
    point_cloud = PointCloud(las_file)
    octree = Octree(point_cloud.points, min_cube_size=1.0, max_depth=5)
    octree.visualize()
