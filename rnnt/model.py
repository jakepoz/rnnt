import torch
import torchaudio


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
    
    def forward(self, mel_features, mel_feature_lens, input_ids, input_id_lens):
        decoder_features, decoder_lengths, decoder_state = self.predictor(input_ids, input_id_lens)

        # Unpad the decoder features not to waste memory with the joint network
        decoder_features = decoder_features[:, :torch.max(decoder_lengths).item(), :]

        # Generate the audio features
        audio_features = self.encoder(mel_features) # (N, C, L)
        audio_features = audio_features.permute(0, 2, 1) # (N, L, C)
        audio_feature_lens = self.encoder.calc_output_lens(mel_feature_lens)

        # Unpad the audio features too
        audio_features = audio_features[:, :torch.max(audio_feature_lens).item(), :]

        # Now, apply the joint model to each combination
        joint_features = self.joint(audio_features, decoder_features)

        # Calculate the loss
        loss = torchaudio.functional.rnnt_loss(logits=joint_features, 
                                            targets=input_ids.int(),
                                            logit_lengths=audio_feature_lens,
                                            target_lengths=input_id_lens.int(), 
                                            blank=-1,
                                            clamp=-1,
                                            reduction="mean")
        
        return loss
    

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
                input_ids = input_ids[:, -1].unsqueeze(0) # Since we have the state, we just need to feed in the last one

                predictor_features, _, predictor_state = self.predictor(input_ids, torch.tensor([len(tokens)], dtype=torch.int64, device=self.device), predictor_state)

                cur_outputs_per_step += 1

        # Return tokens, skipping that first blank which was just to initialize the RNN state
        return tokens[1:] 