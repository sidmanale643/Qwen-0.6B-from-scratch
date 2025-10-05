import torch.nn as nn
import torch

class Config:
    def __init__(self, embedding_dimension=1024, vocabulary_size=151936, n_layers=28, base=1000000.0, max_ctx_length=40960, hidden_dim=3072):  
        self.embed_dim = embedding_dimension
        self.vocab_size = vocabulary_size
        self.n_layers = n_layers
        self.base = base
        self.n_heads = 16
        self.kv_heads = 8
        self.hidden_dim = hidden_dim
        self.head_dim = 128
        self.eps = 1e-6
        self.max_ctx_length = max_ctx_length
        self.dtype = torch.float32
       
config = Config()

def calculate_sin_cos():

    i = torch.arange(0, config.head_dim, 2).float() #S
    freqs = 1.0 / (config.base ** (i / config.head_dim))
    positions = torch.arange(config.max_ctx_length).float()

    angles = positions[:, None] * freqs[None, :]
    # Duplicate angles to match full head_dim for proper RoPE rotation
    angles = torch.cat([angles, angles], dim=1)

    return  torch.sin(angles), torch.cos(angles)
    
def rotate(x, sin, cos):
    batch_size, num_heads, seq_len, head_dim = x.shape

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    rotation_matrix = torch.cat((-x2, x1), dim = -1)

    sin = sin[:seq_len,:].unsqueeze(0).unsqueeze(0)
    cos  = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)

    x_rotated = (x * cos) + (rotation_matrix * sin)

    return x_rotated

class FFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer_1 = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        self.layer_2 = nn.Linear(config.embed_dim, config.hidden_dim , bias=False)
        self.layer_3 = nn.Linear(config.hidden_dim, config.embed_dim, bias=False)
        

    def forward(self, x):
        
        fc_1 = self.layer_1(x)
        fc_2 = self.layer_2(x)
        x = nn.functional.silu(fc_1) * fc_2
        return self.layer_3(x)

class RMSNORM(nn.Module):
    def __init__(self, dim=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Use provided dimension or default to embed_dim
        if dim is None:
            dim = config.embed_dim
        
        self.alpha = nn.Parameter(torch.ones(dim))
        #self.delta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
    
        rms_norm = torch.sqrt(torch.mean(x**2, dim = -1, keepdim= True) + config.eps)
        normed = x / rms_norm
        return self.alpha * normed #+ self.delta

class GQA(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.w_q = nn.Linear(config.embed_dim, config.n_heads * config.head_dim, bias=False)
        self.w_k = nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, bias=False)
        self.w_v = nn.Linear(config.embed_dim, config.kv_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.embed_dim, bias=False)
        
        # Q and K normalization layers
        self.q_norm = RMSNORM(config.n_heads * config.head_dim)
        self.k_norm = RMSNORM(config.kv_heads * config.head_dim)
        
    def forward(self, x, sin, cos):
        b, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # Apply normalization to Q and K
        Q = self.q_norm(Q)
        K = self.k_norm(K)
                
        # Reshape and transpose to (batch, heads, seq_len, head_dim)
        Q = Q.view(b, seq_len, config.n_heads, config.head_dim).transpose(1, 2)
        K = K.view(b, seq_len, config.kv_heads, config.head_dim).transpose(1, 2)
        V = V.view(b, seq_len, config.kv_heads, config.head_dim).transpose(1, 2)
        
        Q = rotate(Q, sin, cos)
        K = rotate(K, sin, cos)

        # Repeat K and V to match Q heads (for grouped attention)
        K = K.repeat_interleave(config.n_heads // config.kv_heads, dim=1)
        V = V.repeat_interleave(config.n_heads // config.kv_heads, dim=1)

        # Compute attention scores
        attention_scores = Q @ K.transpose(-2, -1) / (config.head_dim ** 0.5)
        
        # Create causal mask 
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply mask (set upper triangular to -inf)
        masked_attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax
        # softmax(-inf) = 0
        attention_weights = torch.softmax(masked_attention_scores, dim=-1)
        
        # Apply attention to values
        attention_output = attention_weights @ V
        
        # Reshape back to (batch, seq_len, n_heads * head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(b, seq_len, config.n_heads * config.head_dim)
        
        # Final projection
        output = self.out_proj(attention_output)
        
        return output

class Block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.rms_1 = RMSNORM()
        self.mgqa = GQA()
        self.rms_2 = RMSNORM()
        self.ffn = FFN()

        self.sin, self.cos = calculate_sin_cos()
        
    def forward(self, x):
        
        rms_norm_1 = self.rms_1(x)
        mgqa = self.mgqa(rms_norm_1, self.sin, self.cos)
        mgqa += x

        rms_norm_2 = self.rms_2(mgqa)
        ffn = self.ffn(rms_norm_2)
        ffn += mgqa

        return ffn

class QWEN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim, dtype= config.dtype)
        self.blocks = nn.ModuleList([Block() for i in range(config.n_layers)])
        self.rms_3 = RMSNORM()
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
    def forward(self, x):
        
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)

        x = self.rms_3(x)

        final = self.lm_head(x)

        return final


if __name__ == "__main__":
    qwen = QWEN()
    	
    total_params = sum(p.numel() for p in qwen.parameters())
    print(f"Total number of parameters: {total_params:,}")

    # Account for weight tying between embeddings and lm_head
    total_params_normalized = total_params - qwen.embeddings.weight.numel()
    print(f"Total number of unique parameters: {total_params_normalized:,}")

    print(qwen)
    print(qwen(torch.tensor([1, 2, 3]).unsqueeze(0)))





