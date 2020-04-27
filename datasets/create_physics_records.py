#!/usr/bin/env python3
"""This script creates compressed records for training the network"""
import os
import sys
import json
import argparse
import numpy as np
from glob import glob

from create_physics_scenes import PARTICLE_RADIUS
from physics_data_helper import *


def create_scene_files(scene_dir, scene_id, outfileprefix, splits=16):

    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
        scene_dict = json.load(f)

    box, box_normals = numpy_from_bgeo(os.path.join(scene_dir, 'box.bgeo'))

    partio_dir = os.path.join(scene_dir, 'partio')
    fluid_ids = get_fluid_ids_from_partio_dir(partio_dir)
    num_fluids = len(fluid_ids)
    fluid_id_bgeo_map = {
        k: get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids
    }

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = list(range(len(v)))
        if len(v) != len(frames):
            raise Exception(
                'number of frames for fluid {} ({}) is different from {}'.
                format(k, len(v), len(frames)))

    sublists = np.array_split(frames, splits)

    boring = False  # no fluid and rigid bodies dont move
    last_max_velocities = [1] * 20

    for sublist_i, sublist in enumerate(sublists):
        if boring:
            break

        outfilepath = outfileprefix + '_{0:02d}.msgpack.zst'.format(sublist_i)
        if not os.path.isfile(outfilepath):

            data = []
            for frame_i in sublist:

                feat_dict = {}
                # only save the box for the first frame of each file to save memory
                if frame_i == sublist[0]:
                    feat_dict['box'] = box.astype(np.float32)
                    feat_dict['box_normals'] = box_normals.astype(np.float32)

                feat_dict['frame_id'] = np.int64(frame_i)
                feat_dict['scene_id'] = scene_id

                pos = []
                vel = []
                mass = []
                viscosity = []

                sizes = np.array([0, 0, 0, 0], dtype=np.int32)

                for flid in fluid_ids:
                    bgeo_path = fluid_id_bgeo_map[flid][frame_i]
                    pos_, vel_ = numpy_from_bgeo(bgeo_path)
                    pos.append(pos_)
                    vel.append(vel_)
                    viscosity.append(
                        np.full(pos_.shape[0:1],
                                scene_dict[flid]['viscosity'],
                                dtype=np.float32))
                    mass.append(
                        np.full(pos_.shape[0:1],
                                scene_dict[flid]['density0'],
                                dtype=np.float32))
                    sizes[0] += pos_.shape[0]

                pos = np.concatenate(pos, axis=0)
                vel = np.concatenate(vel, axis=0)
                mass = np.concatenate(mass, axis=0)
                mass *= (2 * PARTICLE_RADIUS)**3
                viscosity = np.concatenate(viscosity, axis=0)

                feat_dict['pos'] = pos.astype(np.float32)
                feat_dict['vel'] = vel.astype(np.float32)
                feat_dict['m'] = mass.astype(np.float32)
                feat_dict['viscosity'] = viscosity.astype(np.float32)

                data.append(feat_dict)

            create_compressed_msgpack(data, outfilepath)


def create_compressed_msgpack(data, outfilepath):
    import zstandard as zstd
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()

    compressor = zstd.ZstdCompressor(level=22)
    with open(outfilepath, 'wb') as f:
        print('writing', outfilepath)
        f.write(compressor.compress(msgpack.packb(data, use_bin_type=True)))


def main():
    parser = argparse.ArgumentParser(
        description=
        "Creates compressed msgpacks for directories with SplishSplash scenes")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to the input directory with the simulation data")
    parser.add_argument(
        "--splits",
        type=int,
        default=16,
        help="The number of files to generate per scene (default=16)")

    args = parser.parse_args()
    os.makedirs(args.output)

    outdir = args.output

    scene_dirs = sorted(glob(os.path.join(args.input, '*')))
    print(scene_dirs)

    for scene_dir in scene_dirs:
        print(scene_dir)
        scene_name = os.path.basename(scene_dir)
        print(scene_name)
        outfileprefix = os.path.join(outdir, scene_name)
        create_scene_files(scene_dir, scene_name, outfileprefix, args.splits)


if __name__ == '__main__':
    main()
