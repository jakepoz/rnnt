import torch
import torch.nn.functional as F

class JointNetwork(torch.nn.Module):
    def __init__(self, audio_features: int, text_features: int, hidden_features: int, num_classes: int):
        super(JointNetwork, self).__init__()
        self.audio_ln = torch.nn.Linear(audio_features, hidden_features)
        self.text_ln = torch.nn.Linear(text_features, hidden_features)

        self.activation = F.tanh
 
        # We used to put dropout in, but not for now during testing
        #self.joint_dropout = torch.nn.Dropout(0.2)
        self.joint_ln = torch.nn.Linear(hidden_features, num_classes)

        self.blank_idx = num_classes - 1

    # Use this version to do training on a batch, where you need to calculate every combination of joint and decoder states
    # Audio Frame expected to be [N, Length, Featuers]
    # Text Frame expected to be [N, Length, Features]
    def forward(self, audio_frame, text_frame):        
        audio_frame = self.audio_ln(audio_frame)
        text_frame = self.text_ln(text_frame)

        audio_frames = audio_frame.unsqueeze(2)
        text_frames = text_frame.unsqueeze(1)

        joint_frames = audio_frames + text_frames

        joint_frames = F.tanh(joint_frames)

        return self.joint_ln(joint_frames)
    
    # Use this version to export the model and just do inference on a single thing at a time
    # Audio Frame expected to be [N, Length, Featuers]
    # Text Frame expected to be [N, Length, Features]
    def single_forward(self, audio_frame, text_frame):
        audio_frame = self.audio_ln(audio_frame)
        text_frame = self.text_ln(text_frame)

        joint_frame = audio_frame + text_frame

        joint_frame = self.activation(joint_frame)

        return self.joint_ln(joint_frame)
        
