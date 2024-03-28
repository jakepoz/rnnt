import torch
import torchaudio
import datasets

from rnnt.util import save_tensor_json



# These functions are meant to fetch the raw datasets from hugging face or some other system
def get_librispeech_dataset(split: str, cache_dir="/media/datasets/librispeech_hf"):
    dataset = datasets.load_dataset("librispeech_asr", cache_dir=cache_dir)
    dataset = dataset[split]

    # This is commented out because all caps results in fewer tokens, and we haven't figured out how to handle > 10k token possibilities yet
    # Map text to lowercase
    # def to_lowercase(example):
    #     example['text'] = example['text'].lower()
    #     return example

    # dataset = dataset.map(to_lowercase)

    return dataset


# This class actually takes the raw audio and text data, applys any augmentations to them, and does the tokenization
class AudioDatasetProcessor(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, featurizer: torch.nn.Module, device: torch.device, audio_augmentation=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.featurizer = featurizer

        self.device = device

        self.audio_augmentation = audio_augmentation

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        audio = torch.from_numpy(row["audio"]["array"]).to(torch.float32).to(self.device)

        if self.audio_augmentation is not None:
            audio = self.audio_augmentation(audio.unsqueeze(-1), sample_rate=row["audio"]["sampling_rate"]).squeeze(-1)

        text = row["text"].lower()

        # with open("sampleaudio.json", "w") as f:
        #         f.write(save_tensor_json(audio.cpu()))

        audio_features = self.featurizer(audio)
        text_tokens = torch.tensor(self.tokenizer.encode(text)).to(self.device)


        return {
            "mel_features": audio_features,
            "input_ids": text_tokens,  
        }
    

class AudioDatasetCollator:
    def __call__(self, batch):
        mel_feature_lens = [x["mel_features"].shape[1] for x in batch]
        input_id_lens = [x["input_ids"].shape[0] for x in batch]

        num_mel_features = batch[0]["mel_features"].shape[0]
        longest_mel_feature = max(mel_feature_lens)

        mel_features = torch.zeros(len(batch), num_mel_features, longest_mel_feature)
        input_ids = torch.zeros(len(batch), max(input_id_lens), dtype=torch.int64)

        for i, x in enumerate(batch):
            mel_features[i, :, :x["mel_features"].shape[1]] = x["mel_features"]
            input_ids[i, :x["input_ids"].shape[0]] = x["input_ids"]

        return {
            "mel_features": mel_features,
            "mel_feature_lens": torch.tensor(mel_feature_lens),
            "input_ids": input_ids,
            "input_id_lens": torch.tensor(input_id_lens),
        }