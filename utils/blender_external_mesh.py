#
# This script replaces the geometry of meshes when changing the frame number.
#
# Run this script once in blender by pressing the "Run Script" button.
# To run the script automatically on loading the blend file enable "Register"
# and save the blend file. This is recommended for rendering animations using
# the command line interface.
#
# Add the following custom properties to a blender object of type mesh to
# automatically replace its mesh when changing the frame
#
# external_files: This is a glob pattern to select files. E.g. //canyon_out/fluid*.npz
#
# start_frame: This is an optional integer. The frame number that corresponds
#              to the first external file.
#
# shade_smooth: This is an optional integer. Set this to a value that is tue
#               to apply smooth shading to the mesh.
#
import bpy
from bpy.app.handlers import persistent
import sys
from glob import glob
import numpy as np


def set_mesh_geometry_from_npz(mesh, path):
    mesh.clear_geometry()
    data = np.load(path, allow_pickle=False)
    vertices = data['pos']
    mesh.from_pydata(vertices, [], [])


def set_mesh_geometry_from_ply(mesh, path):
    try:
        import plyfile
    except ImportError as err:
        print(
            "Cannot import plyfile. Please install this package in blender's python which is in {}"
            .format(sys.exec_prefix))
        raise err

    mesh.clear_geometry()
    plydata = plyfile.PlyData.read(path)
    vertices = plydata['vertex'].data[['x', 'y', 'z']]
    faces = []
    if 'face' in plydata:
        faces = list(plydata['face'].data['vertex_indices'])
    mesh.from_pydata(vertices, [], faces)


def set_mesh_geometry_from_file(mesh, path):
    if path.endswith('.npz'):
        set_mesh_geometry_from_npz(mesh, path)
    elif path.endswith('.ply'):
        set_mesh_geometry_from_ply(mesh, path)
    else:
        raise ValueError('Unsupported file extension')


def load_external_mesh_handler(scene):
    frame = scene.frame_current
    print('frame ', frame)
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'external_files' in obj:
            # skip if not visible
            if obj.hide_viewport and obj.hide_render:
                continue

            external_files = sorted(
                glob(bpy.path.abspath(obj['external_files'])))

            if len(external_files) == 0:
                continue

            start_frame = 0
            if 'start_frame' in obj:
                start_frame = int(obj['start_frame'])

            mesh = obj.data

            # clamp to last idx and use empty mesh for frames before the start idx
            idx = min(len(external_files) - 1, frame - start_frame)
            if idx < 0:
                mesh.clear_geometry()
                obj['current_external'] = ''
                continue

            ply_path = external_files[idx]

            if 'current_external' in obj and obj['current_external'] == ply_path:
                continue

            set_mesh_geometry_from_file(mesh, ply_path)

            if 'shade_smooth' in obj and obj['shade_smooth']:
                values = [True] * len(mesh.polygons)
            else:
                values = [False] * len(mesh.polygons)
            mesh.polygons.foreach_set("use_smooth", values)

            obj['current_external'] = ply_path


def pre_save_external_mesh_handler(dummy):
    # remove properties that we do not want to store
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and 'current_external' in obj:
            del obj['current_external']


bpy.app.handlers.frame_change_pre.append(load_external_mesh_handler)
bpy.app.handlers.save_pre.append(pre_save_external_mesh_handler)
