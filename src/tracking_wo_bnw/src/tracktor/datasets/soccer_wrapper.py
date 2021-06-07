import os.path as osp

import torch
from torch.utils.data import Dataset

from .soccer_sequence import SoccerSequence


class SoccerWrapper(Dataset):
	"""A Wrapper for the MOT_Sequence class to return multiple sequences."""

	def __init__(self, split, dets, dataloader):
		"""Initliazes all subset of the dataset.

		Keyword arguments:
		split -- the split of the dataset to use
		dataloader -- args for the MOT_Sequence dataloader
		"""

		self._data = []
		self._data.append(SoccerSequence('16m_right', 'data/0125-0135/16m_right/img', **dataloader))
		#self._data.append(SoccerSequence('cam1', 'data/0125-0135/cam1/img', **dataloader))
		#self._data.append(SoccerSequence('EPTS_8', 'data/0125-0135/EPTS/8/img', **dataloader))

	def __len__(self):
		return len(self._data)

	def __getitem__(self, idx):
		return self._data[idx]









