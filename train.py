from ultralytics.data.augment import Albumentations
from augmentation import __aug_init__
import argparse
import shutil
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser(
	prog="yolo_train",
	description="Training Yolo Model for forest fire detection",
)

parser.add_argument(
	"-m", "--models",
	nargs='+',
	default=["yolov8s.pt"]
)
parser.add_argument(
	"-e", "--epochs",
	type=int,
	default=200,
)
parser.add_argument(
	"-i", "--imgz",
	type=int,
	default=640,
)
parser.add_argument(
	"-d", "--data",
	type=str,
	default="data.yaml",
)
parser.add_argument(
	"-w", "--workers",
	type=int,
	default=os.cpu_count(),
)
parser.add_argument(
	"-p", "--project-output",
	type=str,
	default="runs"
)


args = parser.parse_args()

Albumentations.__init__ = __aug_init__

MODELS_PRETRAINED = {}

print(f"Number of Workers: {args.workers}")

models = []
for model_name in args.models:
	model = YOLO(model_name)
	if model_name in MODELS_PRETRAINED:
		model.load(MODELS_PRETRAINED[model_name])
	models.append(model)

for i, model in enumerate(models):
	model_name = args.models[i].split('.')[0]

	model.train(
		data=args.data, epochs=args.epochs, imgsz=args.imgz,
		project=args.project_output, name=model_name,
		exist_ok=True,
		workers=args.workers
		batch=32,
		lr0=0.001,
		lrf=0.01,
		warmup_epochs=5,
		warmup_bias_lr=0.00001,
		patience=30,
		optimizer='AdamW',
	)

	shutil.copy2(f"{args.project_output}/{model_name}/weights/best.pt", f"{model_name}_best.pt")
