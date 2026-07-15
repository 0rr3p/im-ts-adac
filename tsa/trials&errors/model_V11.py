import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.

    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return torch.zeros(num_dir, x.size(0), hidden_size).to(device)  ## tolto variable 


###########################################################################
################################ ENCODERS #################################
###########################################################################

class Encoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config['hidden_size_encoder'])

    
    def forward(self, input_data: torch.Tensor):
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        input_encoded = torch.zeros(input_data.size(0), self.seq_len, self.hidden_size).to(device)
    
        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t.squeeze(0)
    
        # Fix: restituisci un tensore di zeri invece di _ o None
        dummy_attentions = torch.zeros(
            input_data.size(0), self.seq_len, self.input_size
        ).to(device)
        return dummy_attentions, input_encoded

class AttnEncoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.add_noise = config['denoising']
        self.directions = config['directions']
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _get_noise(input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        attentions = torch.zeros(input_data.size(0), self.seq_len, self.input_size).to(device) ##tolto variable e aggiunto to_dev
        input_encoded = torch.zeros(input_data.size(0), self.seq_len, self.hidden_size).to(device)

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(device)

        input_data_t = input_data.permute(0, 2, 1).to(device)  # fuori dal loop, calcolato una volta sola e dentro il loop:
        for t in range(self.seq_len):
            x = torch.cat((h_t.expand(self.input_size, -1, -1).permute(1, 0, 2),
                           c_t.expand(self.input_size, -1, -1).permute(1, 0, 2),
                           input_data_t), dim=2)

            
            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size)).to(device)  # (bs, input_size)

            weighted_input = torch.mul(a_t, input_data[:, t, :].to(device))  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t.squeeze(0)
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################





class NonARDecoder(nn.Module):
    def __init__(self, config):
        super(NonARDecoder, self).__init__()
        enc_h = config['hidden_size_encoder']
        dec_h = config['hidden_size_decoder']
        out_f = config['output_size']
        self.seq_len = config['seq_len']

        self.pool = nn.Linear(self.seq_len, 1)
        self.expand = nn.Linear(enc_h, self.seq_len * dec_h)  # ← riespansione learned
        self.dec_h = dec_h

        self.net = nn.Sequential(
            nn.LayerNorm(dec_h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dec_h, out_f)
        )

    def forward(self, input_encoded, y_hist):
        z = self.pool(input_encoded.permute(0, 2, 1)).squeeze(-1)  # (B, enc_h)
        x = self.expand(z).view(-1, self.seq_len, self.dec_h)      # (B, seq_len, dec_h)
        return self.net(x)          

class NonARAttnDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_h  = config['hidden_size_encoder']
        dec_h  = config['hidden_size_decoder']
        out_f  = config['output_size']
        seq_len = config['seq_len']

        # Comprime (B, seq_len, enc_h) → (B, enc_h)
        self.pool = nn.Linear(seq_len, 1)

        # Espande (B, enc_h) → (B, seq_len, dec_h)
        self.expand = nn.Linear(enc_h, seq_len * dec_h)
        self.seq_len = seq_len
        self.dec_h = dec_h

        n_heads = max(1, dec_h // 8)
        self.attn  = nn.MultiheadAttention(dec_h, num_heads=n_heads,
                                            dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dec_h)
        self.norm2 = nn.LayerNorm(dec_h)
        self.ff    = nn.Sequential(
            nn.Linear(dec_h, dec_h * 2),
            nn.GELU(),
            nn.Linear(dec_h * 2, dec_h),
        )
        self.out_proj = nn.Linear(dec_h, out_f)

    def forward(self, input_encoded, y_hist):
        # input_encoded: (B, seq_len, enc_h)
        
        # 1. Comprimi in un vettore, Si ma come non ho idea di come scegliere
        
        
        #z = input_encoded.mean(dim=1)  # (B, enc_h) #-->mean pool
        z = self.pool(input_encoded.permute(0,2,1)).squeeze(-1) #--> bottleneck vettoriale
        
        # 2. Espandi di nuovo alla sequenza: (B, seq_len, dec_h)
        x = self.expand(z).view(-1, self.seq_len, self.dec_h)
        
        # 3. Self-attention (ora su rappresentazione compressa, non su input_encoded diretto)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return self.out_proj(x)

        
######################################################################################################################################################

class AutoEncForecast(nn.Module):
    def __init__(self, config, input_size):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(config, input_size).to(device) if config['input_att'] else \
            Encoder(config, input_size).to(device)
        self.decoder = NonARAttnDecoder(config).to(device) if config['temporal_att'] else NonARDecoder(config).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        if return_attention:
            return outputs, attentions
        return outputs
