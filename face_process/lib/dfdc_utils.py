import json
import os
from glob import glob
from pathlib import Path


def get_original_video_paths(root_dir, basename=False):
    originals = set()
    originals_v = set()
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        dir = Path(json_path).parent
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "REAL":
                original = k
                originals_v.add(original)
                originals.add(os.path.join(dir, original))
    originals = list(originals)
    originals_v = list(originals_v)
    return originals_v if basename else originals


def get_original_with_fakes(root_dir):
    pairs = []
    for json_path in glob(os.path.join(root_dir, "*/metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        for k, v in metadata.items():
            original = v.get("original", None)
            if v["label"] == "FAKE":
                pairs.append((original[:-4], k[:-4] ))

    return pairs


def get_originals_and_fakes(root_dir):
    originals = []
    fakes = []
    for json_path in glob(os.path.join(root_dir, "metadata.json")):
        with open(json_path, "r") as f:
            metadata = json.load(f)
        # i = 0
        for k, v in metadata.items():
            if v["label"] == "FAKE":
                fakes.append([k[:-4], 1, v["split"], v["original"][:-4]])
            else:
                originals.append([k[:-4], 0, v["split"], "none" ])
            # i += 1
            # if i > 1:
            #     break
    return originals, fakes
