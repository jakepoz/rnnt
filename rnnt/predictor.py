# LSTM Based prediction network from
# https://github.com/pytorch/audio/blob/main/src/torchaudio/models/rnnt.py#L392

import torch

from typing import List, Optional, Tuple


class _CustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if ``True``, enables layer normalization. (Default: ``False``)
        layer_norm_epsilon (float, optional):  value of epsilon to use in
            layer normalization layers (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        if layer_norm:
            self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
            self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)
        else:
            self.c_norm = torch.nn.Identity()
            self.g_norm = torch.nn.Identity()

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.

        B: batch size;
        T: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): with shape `(T, B, D)`.
            state (List[torch.Tensor] or None): list of tensors
                representing internal state generated in preceding invocation
                of ``forward``.

        Returns:
            (torch.Tensor, List[torch.Tensor]):
                torch.Tensor
                    output, with shape `(T, B, hidden_dim)`.
                List[torch.Tensor]
                    list of tensors representing internal state generated
                    in current invocation of ``forward``.
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
            c = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            h, c = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate * c + input_gate * cell_gate
            c = self.c_norm(c)
            h = output_gate * c.tanh()
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        state = [h, c]

        return output, state


class LSTMPredictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization
            for LSTM layers. (Default: ``False``)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_hidden_dim: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                _CustomLSTM(
                    symbol_embedding_dim if idx == 0 else lstm_hidden_dim,
                    lstm_hidden_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out
