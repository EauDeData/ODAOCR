import torch
import torch.nn as nn
import math
from torch import Tensor
from vitstr import vitstr_base_patch16_224
from typing import Any, Dict, List, cast
import numpy as np
import torchvision
import os
import json

class GenericModule(torch.nn.Module):
    def __init__(self, module):
        super(GenericModule, self).__init__()
        self.vitstr = module

    def forward(self, x):
        return self.vitstr(x)

        
class ViTAtienzaWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ViTAtienzaWrapper, self).__init__()
        self.module = GenericModule(model)

    def forward(self, x):
        x = x.mean(1).unsqueeze(1)  # Grayscale Image (not true, but will work)
        return self.module(x).permute(1, 0, 2)


class _ProtoModel(torch.nn.Module):
    def __init__(self, model, device, target='totally_padded_image'):
        super(_ProtoModel, self).__init__()
        self.model = model
        self.device = device
        self.target = target

    def forward(self, x):
        output = self.model(x[self.target].to(self.device))
        return {
            'features': output,
            'language_head_output': output
        }

class TransformerDecoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, decoder_token_size, decoder_depth, vocab_size, decoder_width):
        super(TransformerDecoder, self).__init__()

        self.encoder = encoder
        self.projection = torch.nn.Linear(encoder_input_size, decoder_token_size)
        self.memory = torch.nn.Linear(encoder_input_size, decoder_token_size)

        self.gelu_fn = torch.nn.GELU()

        self.layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=decoder_depth)

        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

    def forward(self, X):

        encoder_output = self.encoder(X)['features']  # Pass the batch X through the encoder

        memory = self.memory(encoder_output)

        projected = self.gelu_fn(self.projection(encoder_output))  # Project encoder output to decoder token size
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(projected.size(0)).to(projected.device)

        # Perform decoding using TransformerDecoder
        decoded = self.gelu_fn(self.decoder(
            tgt=projected,
            memory=memory,
            tgt_mask=tgt_mask
        ))


        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)

        return {
            'features': decoded,
            'language_head_output': output,
            'hidden_states': None
        }

class GreedyTextDecoder:
    # Pau's implementation:
    # https://github.com/ptorras/comref-experiments/blob/master/src/core/formatters/ctc/greedy_decoder.py
    """Generate an unpadded token sequence from a CTC output."""

    def __init__(self, confidences: bool = False) -> None:
        """Construct GreedyTextDecoder object."""
        super().__init__()
        self._confidences = confidences

    def __call__(
        self, model_output, blank_index, *args
    ) -> List[Dict[str, Any]]:
        """Convert a model output to a token sequence.

        Parameters
        ----------
        model_output: ModelOutput
            The output of a CTC model. Should contain an output with shape L x B x C,
            where L is the sequence length, B is the batch size and C is the number of
            classes.
        batch: BatchedSample
            Batch information.

        Returns
        -------
        List[Dict[str, Any]]
            A List of sequences of tokens corresponding to the decoded output and the
            output confidences encapsulated within a dictionary.
        """
        ctc_output = model_output["ctc_output"]
        ctc_output = ctc_output.transpose((1, 0, 2))
        indices = ctc_output.argmax(axis=-1)
        output = []

        for sample, mat in zip(indices, ctc_output):
            previous = blank_index
            decoded = []
            confs = []
            for ind, element in enumerate(sample):
                if element == blank_index:
                    previous = blank_index
                    continue
                if element == previous:
                    continue
                decoded.append(element)
                previous = element
                confs.append(mat[ind])

            decoded = np.array(decoded)
            confs = np.array(confs)
            if self._confidences:
                output.append({"text": decoded, "text_conf": confs})
            else:
                output.append({"text": decoded, "text_conf": None})
        return output


class CharTokenizer:
    '''
        This tokenizer may be inputted in our collate FN class so we can put it on the dataloader.
            It's elegant (I think)

    '''

    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
    cls_token = '<CLS>'
    padding_token = '<PAD>'
    ctc_blank = '<BLANK>'

    special_tokens = [bos, eos, unk, cls_token, padding_token, ctc_blank]

    def __init__(self, include_secial=False, local_path='tmp_/tokenizers/', tokenizer_name='tokenizer') -> None:

        os.makedirs(local_path, exist_ok=True)
        self.full_path = os.path.join(local_path, tokenizer_name + '.json')
        if os.path.exists(
                self.full_path
        ):
            print(f'Tokenizer {tokenizer_name} found in {local_path}, loading tokens from local storage.')
            self.tokens = json.load(
                open(self.full_path, 'r')
            )

        else:
            raise NotImplementedError("Hey! The provided path doesn't exist. Download it! Contact amolina@cvc.uab.cat for the full tokenizer implementation")

        self.decode_array = np.array(list(self.tokens.keys()))
        self.include_special = include_secial

    def __len__(self):
        return max(self.tokens.values()) + 1

    def __call__(self, tokens: list) -> np.ndarray:

        return np.array([
            self.tokens[token.lower() if not token in self.special_tokens else token] if (
                                                                                             token.lower() if not token in self.special_tokens else token) in self.tokens else
            self.tokens[self.unk]

            for token in (tokens if not self.include_special
                          else tokens + [self.eos])
        ])

    def decode(self, vector):

        vector = vector.permute(1, 0)
        strings = self.decode_array[vector.numpy()].tolist()

        return [''.join(word) for word in strings]

    def decode_from_numpy_list(self, vector):

        # Vector shaped [BS, SL]
        strings = [self.decode_array[x] for x in vector]
        return [''.join(word) for word in strings]


def prepare_model(vocab_size, device='cuda', decoder_token_size=224, decoder_depth=1, decoder_width=8, 
                  replace_last_layer=False, old_tokenizer_size=None, 
                  load_checkpoint=False, checkpoint_name=None):
    if replace_last_layer: 
        vocab_size_arg = old_tokenizer_size
    else: 
        vocab_size_arg = vocab_size
    
    #### LOAD MODEL ###
    feature_size, model = None, None

    base_model = vitstr_base_patch16_224(pretrained=False)
    base_model_wrapped = ViTAtienzaWrapper(base_model)
    base_model_wrapped.module.vitstr.head = torch.nn.Linear(base_model_wrapped.module.vitstr.head.in_features, 96) 
    base_model_wrapped.module.vitstr.head = torch.nn.Linear(
        base_model_wrapped.module.vitstr.head.in_features,
        base_model_wrapped.module.vitstr.head.in_features
    )
    feature_size = base_model_wrapped.module.vitstr.head.in_features
    base_model_wrapped.module.vitstr.num_classes = feature_size

    model = _ProtoModel(base_model_wrapped, device, target='full_images')
    print('Loaded model with:', len(list(model.parameters())), 'modules.')

    model = TransformerDecoder(model, feature_size, decoder_token_size, decoder_depth, vocab_size_arg, decoder_width)

    if load_checkpoint and checkpoint_name:
        print(f"Loading state dict from: {checkpoint_name}")
        incompatible_keys = model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        print(f"(I, script) Found incompatible keys: {incompatible_keys}")

    
    if replace_last_layer:
        model.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)
        
    model.to(device)
    model.train()
    return model

def clean_special_tokens(string, tokenizer):
    for token in tokenizer.special_tokens:
        string = string.replace(token, '')

    return string

def make_inference(model, tokenizer, decoder, pil_image, device = 'cuda'):
    transforms = torchvision.transforms.Compose((
        lambda x: x.resize((224, 224)),
        torchvision.transforms.PILToTensor(),
        lambda x: (x - x.min()) / max((x.max() - x.min()), 0.01))
    )
    tensor_image = {'full_images': transforms(pil_image)[None, :].to(device)}
    model = model.to(device)

    tokens = model(tensor_image)['language_head_output'].cpu().detach().numpy()
    decoded_tokens = decoder({'ctc_output': tokens}, tokenizer.ctc_blank, None)

    strings = [clean_special_tokens(x.split(tokenizer.eos)[0], tokenizer) for x in
               tokenizer.decode_from_numpy_list([x['text'] for x in decoded_tokens])]


    return strings