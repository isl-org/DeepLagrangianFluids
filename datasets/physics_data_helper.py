"""Helper functions for reading and writing simulation data"""
import os
import re
from glob import glob
import numpy as np


def get_fluid_frame_id_from_bgeo_path(x):
    return int(re.match('.*ParticleData_.+_(\d+)\.bgeo', x).group(1))


def get_fluid_ids_from_partio_dir(partio_dir):
    bgeo_files = glob(os.path.join(partio_dir, 'ParticleData*.bgeo'))
    fluid_ids = set()
    for x in bgeo_files:
        fluid_ids.add(re.match('.*ParticleData_(.+)_\d+\.bgeo', x).group(1))

    return list(sorted(fluid_ids))


def get_fluid_bgeo_files(partio_dir, fluid_id):
    bgeo_files = glob(
        os.path.join(partio_dir, 'ParticleData_{0}_*.bgeo'.format(fluid_id)))
    bgeo_files.sort(key=get_fluid_frame_id_from_bgeo_path)
    return bgeo_files


def numpy_from_bgeo(path):
    import partio
    p = partio.read(path)
    pos = p.attributeInfo('position')
    vel = p.attributeInfo('velocity')
    ida = p.attributeInfo('trackid')  # old format
    if ida is None:
        ida = p.attributeInfo('id')  # new format after splishsplash update
    n = p.numParticles()
    pos_arr = np.empty((n, pos.count))
    for i in range(n):
        pos_arr[i] = p.get(pos, i)

    vel_arr = None
    if not vel is None:
        vel_arr = np.empty((n, vel.count))
        for i in range(n):
            vel_arr[i] = p.get(vel, i)

    if not ida is None:
        id_arr = np.empty((n,), dtype=np.int64)
        for i in range(n):
            id_arr[i] = p.get(ida, i)[0]

        s = np.argsort(id_arr)
        result = [pos_arr[s]]
        if not vel is None:
            result.append(vel_arr[s])
    else:
        result = [pos_arr, vel_arr]

    return tuple(result)


def write_bgeo_from_numpy(outpath, pos_arr, vel_arr):
    import partio

    n = pos_arr.shape[0]
    if not (vel_arr.shape[0] == n and pos_arr.shape[1] == 3 and
            vel_arr.shape[1] == 3):
        raise ValueError(
            "invalid shapes for pos_arr {} and/or vel_arr {}".format(
                pos_arr.shape, vel_arr.shape))

    p = partio.create()
    position_attr = p.addAttribute("position", partio.VECTOR, 3)
    velocity_attr = p.addAttribute("velocity", partio.VECTOR, 3)

    for i in range(n):
        idx = p.addParticle()
        p.set(position_attr, idx, pos_arr[i].astype(float))
        p.set(velocity_attr, idx, vel_arr[i].astype(float))

    partio.write(outpath, p)
