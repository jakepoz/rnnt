import torch


# Just a simple container for all the submodels, so that you can more easily save and load the entire model
class RNNTModel(torch.nn.Module):
    def __init__(self, predictor, encoder, joint):
        super(RNNTModel, self).__init__()
        self.predictor = predictor
        self.encoder = encoder
        self.joint = joint