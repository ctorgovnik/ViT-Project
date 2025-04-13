import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSplitter(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Split into patches
        x = x.view(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        # Reorder dimensions to get [B, num_patches, patch_dim]
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super().__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)
    
    def forward(self, x):
        # x: [B, num_patches, patch_dim]
        return self.linear(x)

class AddPositionEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embedding
    

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
    
    def forward(self, x):
        z = self.norm1(x)
        z = self.attn(z)
        # Residual connection
        x = x + z   
        z = self.norm2(x)
        z = self.mlp(z)
        # Residual connection
        x = x + z
        return x
    

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Calculate dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # 3 for RGB channels
        
        # Patch processing modules
        self.patch_splitter = PatchSplitter(patch_size)
        self.patch_embedding = PatchEmbedding(self.patch_dim, dim)
        
        # Position embeddings
        self.pos_embedding = AddPositionEmbedding(dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        # Split into patches
        x = self.patch_splitter(x)
        
        # Project to embedding dimension
        x = self.patch_embedding(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use class token for classification
        x = x[:, 0]
        
        # Classification head
        x = self.mlp_head(x)
        
        return x