import torch.nn as nn
import torch.nn.functional as F
import torch

class NodeFeatureEncoder(nn.Module):
    def __init__(self, input_steps = 10, d_out=64):
        super(NodeFeatureEncoder, self).__init__()
        
        self.input_dim = input_steps
        
        # Layer definitions: Linear -> ReLU -> LayerNorm
        self.fc1 = nn.Linear(input_steps, d_out)
        self.fc2 = nn.Linear(d_out, d_out)
        self.fc3 = nn.Linear(d_out, d_out)

        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.norm3 = nn.LayerNorm(d_out)

    def forward(self, x):
        """
        Args:
            func_seq: Tensor of shape [batch_size, M, 20], the function values for the past 20 steps.
        
        Returns:
            Tensor of shape [batch_size, M, d_out], the encoded representation of the nodes.
        """
        # x = func_seq.reshape(-1, self.input_dim)  # Flatten to [batch_size, 20 * d_func]
        # print("encoder")
        # Pass through each layer with ReLU + LayerNorm
        # print(x.shape)
        x = self.fc1(x)
        # print("encoder1")
        x = self.norm1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.norm3(x)
        
        return x  # [batch_size, M, d_out]
    
class NodeFeatureDecoder(nn.Module):
    def __init__(self, d_in=64, d_out=1):
        """
        Decoder that maps the encoded node features back to output space.
        Args:
            d_in: Input dimension (from encoder).
            d_out: Output dimension.
        """
        super(NodeFeatureDecoder, self).__init__()
        self.d_in = d_in
        # Layer definitions: Linear -> ReLU -> LayerNorm
        self.fc1 = nn.Linear(d_in, d_in)
        self.fc2 = nn.Linear(d_in, d_in)
        self.fc3 = nn.Linear(d_in, d_out)

        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)
        # self.norm3 = nn.LayerNorm(d_out)

    def forward(self, x, M=100):
        """
        Args:
            node_feat: Tensor of shape [batch_size, M, d_in], the encoded node feature.
        
        Returns:
            Output tensor of shape [batch_size, M, d_out], the decoded node prediction.
        """
        # B, M, d_in = node_feat.shape
        # x = node_feat.view(-1, self.d_in)  # Flatten to [B*M, d_in]

        # Pass through each layer with ReLU + LayerNorm
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # x = self.norm3(x)  
        
        return x  # Reshape back to [batch_size, M, d_out]

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
class MultiHeadFlowAwareAttention(nn.Module):
    def __init__(self, d_in=64, num_heads=8):
        super(MultiHeadFlowAwareAttention, self).__init__()
        self.num_heads = num_heads

        # Per-head MLPs for feature transformation
        self.src_mlps = nn.ModuleList([nn.Linear(d_in, d_in) for _ in range(num_heads)])
        self.dst_mlps = nn.ModuleList([nn.Linear(d_in, d_in) for _ in range(num_heads)])
        
        # Edge encoding: direction (2D) + edge weight (1D)
        self.edge_mlp = nn.Linear(3, d_in)  
    def forward(self, node_feat, Coord, edge_index, edge_weight, M):
        """
        Args:
            node_feat: [batch_size * M, d_in]
            Coord: [batch_size * M, 2]
            edge_index: [2, E * batch_size]
            edge_weight: [E * batch_size, 1]
        Returns:
            A: [batch_size, M, M], right-stochastic attention matrix for each graph in the batch
        """
        batch_size, _, d_in = node_feat.shape[0] // M, node_feat.shape[0], node_feat.shape[1]
        edge_num = M*(M-1)
        src, dst = edge_index
        E = src.shape[0]
   
        h_i = node_feat[src]  # [E * batch_size, d_in]
        h_j = node_feat[dst]  # [E * batch_size, d_in]
      
        delta_coord = Coord[dst] - Coord[src]  # [E * batch_size, 2]
        edge_input = torch.cat([delta_coord, edge_weight], dim=-1)  # [E * batch_size, 3]
        edge_feat = self.edge_mlp(edge_input)  # [E * batch_size, d_in]
       
        attention_scores = []
        for i in range(self.num_heads):
            h_i_mlp = self.src_mlps[i](h_i)  # [E * batch_size, d_in]
            h_j_mlp = self.dst_mlps[i](h_j)  # [E * batch_size, d_in]
            node_score = torch.sum(h_i_mlp * h_j_mlp, dim=-1)  # [E * batch_size]
            edge_score = torch.sum(edge_feat, dim=-1)  # [E * batch_size]
            attention_scores.append(node_score + edge_score)
        attention_scores = torch.stack(attention_scores, dim=-1)  # [E * batch_size, num_heads]
        attention_scores = attention_scores / (d_in ** 0.5)    
        A_raw = torch.zeros(batch_size, M, M, self.num_heads, device=node_feat.device)  # [batch_size, M, M, num_heads]
        for i in range(batch_size):
            start_idx = i * edge_num    
            end_idx = (i + 1) * edge_num
            A_raw[i, src[start_idx:end_idx] - i*M, dst[start_idx:end_idx] - i*M, :] = attention_scores[start_idx:end_idx, :]
        A_softmax = F.softmax(A_raw, dim=2)  # [batch_size, M, M, num_heads]
        A = A_softmax.mean(dim=-1)  # [batch_size, M, M]

        return A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint  

class GraphModel(nn.Module):
    def __init__(self, input_steps=20, horizon = 1, d_in=64, codebook_size = 100, d_out=1, num_heads=8):
        super(GraphModel, self).__init__()
        
        # Define Encoder, Attention, Decoder
        self.encoder = NodeFeatureEncoder(input_steps=input_steps, d_out=d_in)  # Encoder
        self.attention = MultiHeadFlowAwareAttention(d_in=d_in, num_heads=num_heads)  # Attention
        self.decoder = NodeFeatureDecoder(d_in=d_in, d_out=d_out)  # Decoder
        self.M = codebook_size
        self.d_in = d_in
        self.horizon = horizon
        
    def forward(self, data, t_input):
        """
        Args:
            data: A data object containing the graph structure and node features.
            t_input: Time input for ODE solver, shape [num_time_steps]
        
        Returns:
            Output node features, shape [batch_size, M, d_out]
        """
        # Extract the necessary information from data
        x = data.x  # Node features, shape [M*batch_size, d_in]
        edge_index = data.edge_index  # Edge indices, shape [2, E*batch_size]
        edge_weight = data.edge_weight  # Edge weights, shape [E*batch_size, 1]
        Coord = data.Coord   #[M*batch_size, 2]
        node_feat = self.encoder(x)  # [M, d_in]
        A = self.attention(node_feat, Coord, edge_index, edge_weight, self.M)  # [batch_size, M, M]
        node_feat = self.ode_update(A, node_feat, t_input, self.M)  # [horizon, batch_size, M, d_in]
        output = self.decoder(node_feat.permute(1,2,0,3).reshape(-1, self.horizon, self.d_in)).squeeze(-1)  # [batch_size*M, horizon]    
        return output
    
        
    def ode_update(self, A, x, t_input, M):
        """
        Applies the ODE update: dx/dt = (A - I)x using odeint for numerical integration.
        Args:
            A: Influence matrix, shape [batch_size, M, M]
            x: Current node features, shape [batch_size * M, d_in]
            t_input: Time input for ODE solver, shape [num_time_steps]
            M: codebook_size
        Returns:
            Updated node features after ODE integration.
        """
        batch_size, _, d_in = x.shape[0] // M, x.shape[0], x.shape[1] 
        x = x.view(batch_size, M, d_in)  # [batch_size, M, d_in]
        # Define the ODE model: dx/dt = (A - I)x
        def ode_func(t, x):
            A_batch = A  # Shape: [batch_size, M, M]
            # Compute (A - I) * x for each batch
            return torch.matmul(A_batch - torch.eye(M, device=A.device), x)  # [batch_size, M, d_in]
        x_new = odeint(ode_func, x, t_input, method='rk4')  # [num_time_steps, batch_size, M, d_in]
        return x_new