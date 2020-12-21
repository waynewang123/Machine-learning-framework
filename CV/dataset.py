import torch 
import numpy as np 
from PIL import Image
from PIL import ImageFile

# this will care of these corrupt images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class classificationDataset:
	'''
	A general classification dataset clas that you can use for all kinds of image classification
	problems. such as binary classification, multi-class, multi-label classification
	'''
	def __init__(
		self,
		image_paths,
		targets,
		resize = None,
		augumentations = None
		):
		'''
		:param image_paths: list of path to images
		:param targets: numpy array
		:param resize: tuple, e.g (256,256),resize image if not None
		:param augmentations: albumentation augumentations
	
		'''
		self.image_paths = image_paths
		self.targets = targets
		self.resize = resize 
		self.augumentations = augumentations

	def __len__(self):
		"""
		Return the total number of samples in the dataset
		"""
		return len(self.image_paths)

	def __getitem__(self,item):
		"""
		for a given 'item' index, return everything we need to train a given model
		"""
		#use PIL to open the image
		image = Image.open(self.image_paths[item])
		# convert image to rgb, we have single channel images 
		image = image.convert('RGB')
		targets = self.targets[item]

		if self.resize is not None:
			image = image.resize(
				(self.resize[1], self.resize[0]),
				resample = Image.BILINEAR
				)
		image = np.array(image)
		# if we have albumentation augmentations # add them to the image
		if self.augmentations is not None:
			augmented = self.augmentations(image=image) 
			image = augmented["image"]
		# pytorch expects CHW instead of HWC
		image = np.transpose(image, (2, 0, 1)).astype(np.float32)
		# return tensors of image and targets
		# take a look at the types!
		# for regression tasks,
		# dtype of targets will change to torch.float 
		return {
			"image": torch.tensor(image, dtype=torch.float),
			"targets": torch.tensor(targets, dtype=torch.long), 
			}



