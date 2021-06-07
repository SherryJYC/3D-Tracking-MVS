
from torch.utils.data import ConcatDataset



from .soccer_wrapper import SoccerWrapper



class Datasets(object):
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets, *args):
        """Initialize the corresponding dataloader.

        Keyword arguments:
        dataset --  the name of the dataset
        args -- arguments used to call the dataloader
        """



        if len(args) == 0:
            args = [{}]

        
        self.datasets = SoccerWrapper("16m_right", "16m_right", *args)
        #self.datasets = SoccerWrapper("cam1", "cam1", *args)
        #self.datasets = SoccerWrapper("EPTS_8", "EPTS_8", *args)



    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]
