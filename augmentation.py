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
			# Simulate lighting conditions and smoke effects
			A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
			A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=30, p=0.5),
			A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.2, p=0.3),
	
			# Simulate drone movement and sensor noise
			A.MotionBlur(blur_limit=(2, 5), p=0.3),
			A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

			# Simulate occlusion and perspective shifts
			A.CoarseDropout(max_holes=4, max_height=4, max_width=4, fill_value=0, p=0.2),
			A.Perspective(scale=(0.03, 0.07), p=0.2),

			# Simulate focus issues
			A.OneOf([
				A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=1.0),
				A.Blur(blur_limit=2, p=1.0),
			], p=0.3),

			# Simulate shadows from smoke plumes and compression artifacts
			A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.2),
			A.ImageCompression(quality_lower=85, p=0.2),
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

