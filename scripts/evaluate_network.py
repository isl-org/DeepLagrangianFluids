#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_val
from fluid_evaluation_helper import FluidErrors


def evaluate_tf(model, val_dataset, frame_skip, fluid_errors=None, scale=1):
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

                fluid_errors.add_errors(scene_id, frame0_id, frame1_id, scale*pr_pos1,
                                        scale*gt_pos1)
                fluid_errors.add_errors(scene_id, frame0_id, frame2_id, scale*pr_pos2,
                                        scale*gt_pos2)

            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])

    print(result)
    print('done')

    return result


def evaluate_whole_sequence_tf(model,
                               val_dataset,
                               frame_skip,
                               fluid_errors=None,
                               scale=1):
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
                                    scale*pr_pos,
                                    scale*gt_pos,
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


def evaluate_torch(model, val_dataset, frame_skip, device, fluid_errors=None, scale=1):
    import torch
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
                box = torch.from_numpy(frames[0]['box'][0]).to(device)
                box_normals = torch.from_numpy(
                    frames[0]['box_normals'][0]).to(device)
                gt_pos1 = frames[1]['pos0'][0]
                gt_pos2 = frames[2]['pos0'][0]

                inputs = (torch.from_numpy(frames[0]['pos0'][0]).to(device),
                          torch.from_numpy(frames[0]['vel0'][0]).to(device),
                          None, box, box_normals)
                pr_pos1, pr_vel1 = model(inputs)

                inputs = (pr_pos1, pr_vel1, None, box, box_normals)
                pr_pos2, pr_vel2 = model(inputs)

                fluid_errors.add_errors(scene_id, frame0_id, frame1_id,
                                        scale*pr_pos1.cpu().detach().numpy(), scale*gt_pos1)
                fluid_errors.add_errors(scene_id, frame0_id, frame2_id,
                                        scale*pr_pos2.cpu().detach().numpy(), scale*gt_pos2)

            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])

    print(result)
    print('done')

    return result


def evaluate_whole_sequence_torch(model,
                                  val_dataset,
                                  frame_skip,
                                  device,
                                  fluid_errors=None,
                                  scale=1):
    import torch
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
            box = torch.from_numpy(data['box'][0]).to(device)
            box_normals = torch.from_numpy(data['box_normals'][0]).to(device)
            init_pos = torch.from_numpy(data['pos0'][0]).to(device)
            init_vel = torch.from_numpy(data['vel0'][0]).to(device)

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
                                    scale*pr_pos.cpu().numpy(),
                                    scale*gt_pos,
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


def eval_checkpoint(checkpoint_path, val_files, fluid_errors, options, cfg):
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    if checkpoint_path.endswith('.index'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model',{}))
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        checkpoint.restore(
            os.path.splitext(checkpoint_path)[0]).expect_partial()

        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors, **cfg.get('evaluation',{}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation',{}))
    elif checkpoint_path.endswith('.h5'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model',{}))
        model.init()
        model.load_weights(checkpoint_path, by_name=True)
        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors, **cfg.get('evaluation',{}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation',{}))
    elif checkpoint_path.endswith('.pt'):
        import torch

        model = trainscript.create_model(**cfg.get('model',{}))
        checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.to(options.device)
        model.requires_grad_(False)
        evaluate_torch(model, val_dataset, options.frame_skip, options.device,
                       fluid_errors, **cfg.get('evaluation',{}))
        evaluate_whole_sequence_torch(model, val_dataset, options.frame_skip,
                                      options.device, fluid_errors, **cfg.get('evaluation',{}))
    else:
        raise Exception('Unknown checkpoint format')


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
    parser = argparse.ArgumentParser(
        description="Evaluates a fluid network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainscript",
                        type=str,
                        required=True,
                        help="The python training script.")
    parser.add_argument("--cfg", type=str, required=True, help="The path to the yaml config file")
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        required=False,
        help="The checkpoint iteration. The default is the last checkpoint.")
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="If set uses the specified weights file instead of a checkpoint.")
    parser.add_argument("--frame-skip",
                        type=int,
                        default=5,
                        help="The frame skip. Default is 5.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    global trainscript
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript = importlib.import_module(module_name)

    train_dir = module_name + '_' + os.path.splitext(os.path.basename(args.cfg))[0]
    val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))

    if args.weights is not None:
        print('evaluating :', args.weights)
        output_path = args.weights + '_eval.json'
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', args.weights, output_path)
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            fluid_errors = FluidErrors()
            eval_checkpoint(args.weights, val_files, fluid_errors,
                            args, cfg)
            fluid_errors.save(output_path)
    else:
        # get a list of checkpoints

        # tensorflow checkpoints
        checkpoint_files = glob(
            os.path.join(train_dir, 'checkpoints', 'ckpt-*.index'))
        # torch checkpoints
        checkpoint_files.extend(
            glob(os.path.join(train_dir, 'checkpoints',
                              'ckpt-*.pt')))
        all_checkpoints = sorted([
            (int(re.match('.*ckpt-(\d+)\.(pt|index)', x).group(1)), x)
            for x in checkpoint_files
        ])

        # select the checkpoint
        if args.checkpoint_iter is not None:
            checkpoint = dict(all_checkpoints)[args.checkpoint_iter]
        else:
            checkpoint = all_checkpoints[-1]

        output_path = train_dir + '_eval_{}.json'.format(checkpoint[0])
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', checkpoint, output_path)
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            print('evaluating :', checkpoint)
            fluid_errors = FluidErrors()
            eval_checkpoint(checkpoint[1], val_files, fluid_errors,
                            args, cfg)
            fluid_errors.save(output_path)

    print_errors(fluid_errors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
