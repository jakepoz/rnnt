import torch


# Just a simple container for all the submodels, so that you can more easily save and load the entire model
class RNNTModel(torch.nn.Module):
    def __init__(self, predictor, encoder, joint):
        super(RNNTModel, self).__init__()
        self.predictor = predictor
        self.encoder = encoder
        self.joint = joint

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def greedy_decode(self, mel_features: torch.Tensor, mel_feature_lens: torch.Tensor, max_length: int = 200) -> list[int]:
        assert mel_features.shape[0] == 1, "Greedy decoding only works with a batch size of 1"

        audio_features = self.encoder(mel_features) # (N, C, L)
        audio_features = audio_features.permute(0, 2, 1) # (N, L, C)
        audio_feature_lens = self.encoder.calc_output_lens(mel_feature_lens)

        tokens = [self.joint.blank_idx]

        cur_audio_time = 0
        max_audio_time = audio_features.shape[1]

        cur_outputs_per_step = 0
        max_outputs_per_step = 10

        input_ids = torch.tensor([tokens], dtype=torch.int64, device=self.device)
        predictor_features, _, predictor_state = self.predictor(input_ids, torch.tensor([len(tokens)], dtype=torch.int64, device=self.device))

        while cur_audio_time < max_audio_time and len(tokens) < max_length:
            
            joint_features = self.joint.single_forward(audio_features[:, cur_audio_time, :], predictor_features[:, -1, :])

            # Get the most likely token
            token_idx = joint_features.argmax(dim=-1).item()

            
            if token_idx == self.joint.blank_idx or cur_outputs_per_step >= max_outputs_per_step:
                cur_audio_time += 1
                cur_outputs_per_step = 0
            else:
                tokens.append(token_idx)

                # Update the predictor features
                input_ids = torch.tensor([tokens], dtype=torch.int64, device=self.device)
                predictor_features, _, predictor_state = self.predictor(input_ids, torch.tensor([len(tokens)], dtype=torch.int64, device=self.device), predictor_state)

                cur_outputs_per_step += 1

        # Convert the token ids back to text via the tokenmap
        return tokens[1:] 