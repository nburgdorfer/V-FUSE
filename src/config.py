import yaml
import os


def save_config(output_path, cfg):
    with open(os.path.join(output_path, "config.yaml"), 'w') as config_file:
        yaml.dump(cfg, config_file)


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get('inherit_from')
    if inherit_from is not None:
        cfg = load_config(inherit_from)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_bounding_box(cfg, bb_path):
    with open(bb_path, 'r') as f:
        bb_config = yaml.full_load(f)

    update_recursive(cfg, bb_config)


def load_scene_list(scenes_list_file):
    with open(scenes_list_file,'r') as sf:
        scenes = sf.readlines()
    scenes = [s.strip() for s in scenes]

    return scenes


def load_invalid_frames(invalid_frames_file):
    with open(invalid_frames_file, 'r') as iff:
        invalid_frames = yaml.full_load(iff)
    return invalid_frames
