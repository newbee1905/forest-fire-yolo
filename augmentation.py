from ultralytics.utils import LOGGER, colorstr

def __aug_init__(self, p=1.0):
	self.p = p
	self.transform = None
	prefix = colorstr("albumentations: ")

	try:
		import albumentations as A

		# List of possible spatial transforms
		spatial_transforms = {
			"Affine",
			"BBoxSafeRandomCrop",
			"CenterCrop",
			"CoarseDropout",
			"Crop",
			"CropAndPad",
			"CropNonEmptyMaskIfExists",
			"D4",
			"ElasticTransform",
			"Flip",
			"GridDistortion",
			"GridDropout",
			"HorizontalFlip",
			"Lambda",
			"LongestMaxSize",
			"MaskDropout",
			"MixUp",
			"Morphological",
			"NoOp",
			"OpticalDistortion",
			"PadIfNeeded",
			"Perspective",
			"PiecewiseAffine",
			"PixelDropout",
			"RandomCrop",
			"RandomCropFromBorders",
			"RandomGridShuffle",
			"RandomResizedCrop",
			"RandomRotate90",
			"RandomScale",
			"RandomSizedBBoxSafeCrop",
			"RandomSizedCrop",
			"Resize",
			"Rotate",
			"SafeRotate",
			"ShiftScaleRotate",
			"SmallestMaxSize",
			"Transpose",
			"VerticalFlip",
			"XYMasking",
		}	# from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

		T = [
			A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), # Simulate changes in lighting conditions
			A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.5), # Enhanced color shifts due to smoke and fire
			A.GaussNoise(var_limit=(10.0, 60.0), p=0.3), # Simulate sensor noise
			A.MotionBlur(blur_limit=(3, 7), p=0.4),	# Simulate drone movement
			A.Rotate(limit=30, p=0.5),	# Simulate drone tilt
			A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.7, alpha_coef=0.2, p=0.4), # Simulate light smoke
			A.RandomBrightnessContrast(brightness_limit=(0.0, 0.5), contrast_limit=0.2, p=0.3), # Simulate fire illumination
			A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.2), # Simulate partially obscured views
			A.Perspective(scale=(0.05, 0.1), p=0.3), # Simulate changes in drone angle
			A.OneOf([
				A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
				A.Blur(blur_limit=3, p=1.0),
			], p=0.3), # Simulate focus issues
			A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.3), # Simulate shadows from smoke plumes
			A.ImageCompression(quality_lower=75, p=0.2),
		]

		# Compose transforms
		self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
		self.transform = (
			A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
			if self.contains_spatial
			else A.Compose(T)
		)
		LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
	except ImportError:	# package not installed, skip
		pass
	except Exception as e:
		LOGGER.info(f"{prefix}{e}")

