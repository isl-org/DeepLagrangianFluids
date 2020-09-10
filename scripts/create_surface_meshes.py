#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))

from multiprocessing import Pool


def read_particles(filename):
    data = np.load(filename, allow_pickle=False)
    return data['pos']


def particles_to_mesh(particles, particle_radius, voxel_size, level):
    """creates a density grid and then extracts the levelset as quad mesh"""
    if not particle_radius >= voxel_size:
        raise ValueError("particle_radius ({}) >= voxel_size ({})".format(
            particle_radius, voxel_size))
    if not voxel_size > 0:
        raise ValueError("voxel_size ({}) > 0".format(voxel_size))
    if not level > 0:
        raise ValueError("level ({}) > 0".format(level))
    import pyopenvdb as vdb
    from scipy.spatial import cKDTree

    def kernel(sqr_d, sqr_h):
        return np.maximum(0, (1 - sqr_d / sqr_h)**3)

    point_radius = particle_radius / voxel_size
    points = particles / voxel_size
    tree = cKDTree(points)

    grid = vdb.FloatGrid()
    accessor = grid.getAccessor()

    visited_indices = set()
    unvisited_indices = set()
    visited_points = np.zeros(shape=points.shape[0:1], dtype=np.uint8)

    def compute_value(ijk):
        nn = tree.query_ball_point(ijk, point_radius)
        if nn:
            sqr_dist = np.sum((points[nn] - np.asarray(ijk))**2, axis=-1)
            # use negative values to get normals pointing outwards
            value = -np.sum(kernel(sqr_dist, point_radius**2))
            visited_points[nn] = 1
        else:
            value = 0
        accessor.setValueOn(ijk, value=value)
        return value

    neighbors = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
                         dtype=np.int32)

    while np.count_nonzero(visited_points) != len(visited_points):
        unvisited_points_mask = visited_points == 0
        idx = tuple(np.round(points[unvisited_points_mask][0]).astype(np.int32))
        visited_points[np.argwhere(unvisited_points_mask)[0]] = 1
        unvisited_indices.add(idx)

        while len(unvisited_indices):
            idx = unvisited_indices.pop()
            visited_indices.add(idx)
            value = compute_value(idx)
            if value:
                new_indices = neighbors + idx
                for x in new_indices:
                    i = tuple(x)
                    if not i in visited_indices:
                        unvisited_indices.add(i)

    # we use negative values to get normals pointing outwards
    vertices, quads = grid.convertToQuads(isovalue=-level)
    vertices *= voxel_size

    return vertices, quads


def write_quadmesh_ply(path, vertices, quads):
    import plyfile
    verts_ = np.empty((vertices.shape[0],),
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    verts_['x'] = vertices[:, 0]
    verts_['y'] = vertices[:, 1]
    verts_['z'] = vertices[:, 2]

    quads_ = np.empty((quads.shape[0],), dtype=[('vertex_indices', 'i4', (4,))])
    for i in range(quads.shape[0]):
        quads_[i] = (tuple(quads[i]),)

    el_verts = plyfile.PlyElement.describe(
        verts_,
        'vertex',
    )
    el_faces = plyfile.PlyElement.describe(
        quads_,
        'face',
    )
    plyfile.PlyData([el_verts, el_faces]).write(path)


def create_mesh(input_file, idx, args):
    xyz = read_particles(input_file)

    vertices, quads = particles_to_mesh(particles=xyz,
                                        particle_radius=args.particle_radius,
                                        voxel_size=args.voxel_size,
                                        level=args.level)

    outpath = os.path.join(args.outdir,
                           args.outfileprefix + '{0:04d}.ply'.format(idx))

    write_quadmesh_ply(outpath, vertices, quads)
    print(input_file, '->', outpath)


def main():
    parser = argparse.ArgumentParser(
        description="Creates a mesh sequence from a particle ply sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_glob",
                        type=str,
                        required=True,
                        help="The input glob, e.g. 'mydata/fluid*.npz'")
    parser.add_argument("--outdir",
                        type=str,
                        required=True,
                        help="The output dir for the surface mesh ply files")
    parser.add_argument("--outfileprefix",
                        type=str,
                        default='levelset_mesh_',
                        help="The output file prefix")
    parser.add_argument("--particle_radius",
                        type=float,
                        default=0.1,
                        help="The particle radius for the density computation")
    parser.add_argument("--voxel_size",
                        type=float,
                        default=0.03,
                        help="The voxel size for the density grid")
    parser.add_argument(
        "--level",
        type=float,
        default=0.5,
        help="The density value at which to extract the level set")
    parser.add_argument(
        "--ncpu",
        type=int,
        default=-1,
        help="The number of processes to use. Default is -1 which means auto")

    args = parser.parse_args()

    input_files = sorted(glob(args.input_glob))

    for input_file in input_files:
        if not os.path.isfile(input_file):
            raise ValueError('{} is not a file.'.format(input_file))
    print(len(input_files), 'files')

    arguments = []
    for i, x in enumerate(input_files):
        arguments.append((x, i, args))

    processes = None
    if args.ncpu >= 1:
        processes = args.ncpu
    with Pool(processes) as pool:
        pool.starmap(create_mesh, arguments)

    return 0


if __name__ == '__main__':
    sys.exit(main())
