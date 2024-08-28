import yaml
import os
from os import path

config = {
	"path": os.getcwd(),
  "names": ["Fire", "Smoke"],
  "nc": 2,
  "test": [],
  "train": [],
  "val": [],
}

for dataset in os.listdir("datasets"):
	location = f"datasets/{dataset}"
	if path.isdir(location):
		config["test"].append(f"{location}/test/images")
		config["val"].append(f"{location}/valid/images")
		config["train"].append(f"{location}/train/images")

with open('data.yaml', 'w+') as f:
  yaml.dump(config, f)
