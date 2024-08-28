from dotenv import load_dotenv
from roboflow import Roboflow
import os

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

# These datasets using
# - Fire = 0
# - Smoke = 1
datasets = {
	"forest-fire-detection-lucg0": {
		"workspace": "dongguk-nr7gx",
		"version": 6,
	},
	"forest-fire-and-smoke-nw1yo": {
		"workspace": "master-candidate",
		"version": 5,
	},
	"wildfire-detection-with-bounding-boxes-d-fire-smoke-only": {
		"workspace": "unlv-c6san",
		"version": 1,
	},
}

for project, info in datasets.items():
	dataset = rf.workspace(info["workspace"]).project(project).version(info["version"]).download("yolov8")
