#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_val
from fluid_evaluation_helper import FluidErrors


def evaluate(model, val_dataset, frame_skip, fluid_errors=None):
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = 0
    frames = []
    for data in val_dataset:
        if data['frame_id0'][0] == 0:
            frames = []
        if data['frame_id0'][0] % skip < 3:
            frames.append(data)
        if data['frame_id0'][0] % skip == 3:

            if len(
                    set([
                        frames[0]['scene_id0'][0], frames[1]['scene_id0'][0],
                        frames[2]['scene_id0'][0]
                    ])) == 1:
                scene_id = frames[0]['scene_id0'][0]
                if last_scene_id != scene_id:
                    last_scene_id = scene_id
                    print(scene_id, end=' ', flush=True)
                frame0_id = frames[0]['frame_id0'][0]
                frame1_id = frames[1]['frame_id0'][0]
                frame2_id = frames[2]['frame_id0'][0]
                box = frames[0]['box'][0]
                box_normals = frames[0]['box_normals'][0]
                gt_pos1 = frames[1]['pos0'][0]
                gt_pos2 = frames[2]['pos0'][0]

                inputs = (frames[0]['pos0'][0], frames[0]['vel0'][0], None, box,
                          box_normals)
                pr_pos1, pr_vel1 = model(inputs)

                inputs = (pr_pos1, pr_vel1, None, box, box_normals)
                pr_pos2, pr_vel2 = model(inputs)

                fluid_errors.add_errors(scene_id, frame0_id, frame1_id, pr_pos1,
                                        gt_pos1)
                fluid_errors.add_errors(scene_id, frame0_id, frame2_id, pr_pos2,
                                        gt_pos2)

            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])

    print(result)
    print('done')

    return result


def evaluate_whole_sequence(model, val_dataset, frame_skip, fluid_errors=None):
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = None
    for data in val_dataset:
        scene_id = data['scene_id0'][0]
        if last_scene_id is None or last_scene_id != scene_id:
            print(scene_id, end=' ', flush=True)
            last_scene_id = scene_id
            box = data['box'][0]
            box_normals = data['box_normals'][0]
            init_pos = data['pos0'][0]
            init_vel = data['vel0'][0]

            inputs = (init_pos, init_vel, None, box, box_normals)
        else:
            inputs = (pr_pos, pr_vel, None, box, box_normals)

        pr_pos, pr_vel = model(inputs)

        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % skip == 0:
            gt_pos = data['pos0'][0]
            fluid_errors.add_errors(scene_id,
                                    0,
                                    frame_id,
                                    pr_pos,
                                    gt_pos,
                                    compute_gt2pred_distance=True)

    result = {}
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])

    print(result)
    print('done')

    return result


def eval_checkpoint(checkpoint_path, val_files, fluid_errors, options):
    import tensorflow as tf

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    model = trainscript.create_model()
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
    checkpoint.restore(checkpoint_path).expect_partial()

    evaluate(model, val_dataset, options.frame_skip, fluid_errors)
    evaluate_whole_sequence(model, val_dataset, options.frame_skip,
                            fluid_errors)


def print_errors(fluid_errors):
    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])
    print('====================\n', result)


def main():
    parser = argparse.ArgumentParser(description="Evaluates a fluid network")
    parser.add_argument("--trainscript",
                        type=str,
                        required=True,
                        help="The python training script.")
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        required=False,
        help="The checkpoint iteration. The default is the last checkpoint.")
    parser.add_argument("--frame-skip",
                        type=int,
                        default=5,
                        help="The frame skip. Default is 5.")

    args = parser.parse_args()

    global trainscript
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript = importlib.import_module(module_name)

    # get a list of checkpoints
    checkpoint_files = glob(
        os.path.join(trainscript.train_dir, 'checkpoints', 'ckpt-*.index'))
    all_checkpoints = sorted([(int(re.match('.*ckpt-(\d+)\.index', x).group(1)),
                               os.path.splitext(x)[0])
                              for x in checkpoint_files])

    # select the checkpoint
    if args.checkpoint_iter is not None:
        checkpoint = dict(all_checkpoints)[args.checkpoint_iter]
    else:
        checkpoint = all_checkpoints[-1]

    output_path = args.trainscript + '_eval_{}.json'.format(checkpoint[0])
    if os.path.isfile(output_path):
        print('Printing previously computed results for :', checkpoint)
        fluid_errors = FluidErrors()
        fluid_errors.load(output_path)
    else:
        print('evaluating :', checkpoint)
        fluid_errors = FluidErrors()
        eval_checkpoint(checkpoint[1], trainscript.val_files, fluid_errors,
                        args)
        fluid_errors.save(output_path)

    print_errors(fluid_errors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
