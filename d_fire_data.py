import yaml
import gdown
import zipfile
import shutil
from os import path
import os
import pandas as pd

dataset_id = "19LSrZHYQqJSdKgH8Mtlgg7-i-L3eRhbh"

output_dir = "D-Fire"
output = f"{output_dir}/d-fire.zip"

os.makedirs(output_dir, exist_ok=True)

gdown.download(id=dataset_id, output=output, quiet=False)

with zipfile.ZipFile(output, "r") as zip_ref:
	zip_ref.extractall(output_dir)

test_dir = f"{output_dir}/test"
valid_dir = f"{output_dir}/valid"

shutil.move(test_dir, valid_dir)

os.makedirs(test_dir, exist_ok=True)
os.makedirs(f"{test_dir}/labels", exist_ok=True)
os.makedirs(f"{test_dir}/images", exist_ok=True)

for subfolder in ["train", "test", "valid"]:
	labels_path = f"{output_dir}/{subfolder}/labels"
	for label_filename in os.listdir(labels_path):
		label_path = f"{labels_path}/{label_filename}"
		try:
			df = pd.read_csv(label_path, sep=" ", header=None)
			df.iloc[:, 0] ^= 1
			df.to_csv(label_path, sep=" ", header=False, index=False, float_format="%.6f")
		except:
			pass

d_fire_config = {
  "names": ["Fire", "Smoke"],
  "nc": 2,
  "test": f"../test/images",
  "train": f"./train/images",
  "val": f"./valid/images",
}

with open(f"{output_dir}/data.yaml", "w+") as f:
	yaml.dump(d_fire_config, f)
