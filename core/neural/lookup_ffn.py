import torch
import torch.nn as nn


class LookupFFN(nn.Module):
    """
    LookupFFN: Memory-efficient Feed Forward Network replacement

    This implementation replaces GEMM operations with memory lookups,
    optimized for CPU inference on Intel Xeon E5-2640 processors.
    Based on "LookupFFN: Making Transformers Compute-lite for CPU inference"
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_centroids: int = 1024,
        activation: str = "gelu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize LookupFFN module

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (typically 4x input_dim in transformers)
            output_dim: Output dimension (typically same as input_dim)
            num_centroids: Number of centroids for vector quantization
            activation: Activation function ("gelu", "relu", "silu")
            dtype: Data type for weights
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_centroids = num_centroids
        self.dtype = dtype

        # Declare activation attribute with broader type
        self.activation: nn.Module

        # Initialize centroid tables
        self.register_buffer(
            "input_centroids", torch.randn(num_centroids, input_dim, dtype=dtype) * 0.02
        )
        self.register_buffer(
            "hidden_centroids",
            torch.randn(num_centroids, hidden_dim, dtype=dtype) * 0.02,
        )

        # Initialize lookup tables (these replace the weight matrices)
        self.register_buffer(
            "lookup_table_fc1",
            torch.randn(num_centroids, hidden_dim, dtype=dtype) * 0.02,
        )
        self.register_buffer(
            "lookup_table_fc2",
            torch.randn(num_centroids, output_dim, dtype=dtype) * 0.02,
        )

        # Bias terms
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_dim, dtype=dtype))
        self.fc2_bias = nn.Parameter(torch.zeros(output_dim, dtype=dtype))

        # Set activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # LSH parameters for faster nearest neighbor search
        self.lsh_num_tables = 8
        self.lsh_hash_size = 16
        self.register_buffer(
            "lsh_proj_matrices",
            torch.randn(
                self.lsh_num_tables, self.lsh_hash_size, input_dim, dtype=dtype
            ),
        )
        self.register_buffer(
            "lsh_centroids",
            torch.zeros(
                self.lsh_num_tables,
                2**self.lsh_hash_size,
                num_centroids,
                dtype=torch.long,
            ),
        )
        self._build_lsh_tables()

    def _build_lsh_tables(self):
        """
        Build LSH tables for faster nearest centroid lookup
        This is crucial for performance on CPUs
        """
        # Compute hash codes for each centroid
        for table_idx in range(self.lsh_num_tables):
            proj_matrix = self.lsh_proj_matrices[table_idx]
            # Project centroids
            projections = torch.matmul(self.input_centroids, proj_matrix.t())
            # Create hash codes (1 bit per projection)
            hash_codes = (projections > 0).int()
            # Convert bit array to integers
            hash_values = torch.zeros(self.num_centroids, dtype=torch.long)
            for i in range(self.lsh_hash_size):
                hash_values = hash_values | (hash_codes[:, i].long() << i) # type: ignore [index, operator]

            # Add centroids to hash buckets
            for centroid_idx in range(self.num_centroids):
                hash_val = hash_values[centroid_idx].item() # type: ignore [index]
                self.lsh_centroids[table_idx, hash_val, centroid_idx] = 1 # type: ignore [index]

    def _find_nearest_centroids(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find nearest centroids to input vectors using LSH

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Indices of nearest centroids of shape [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.input_dim)

        # Use LSH for candidate selection
        candidate_centroids = set()
        for i in range(batch_size * seq_len):
            candidates_for_vector = set()
            vector = x_flat[i]

            # Get hash values for this vector across all tables
            for table_idx in range(self.lsh_num_tables):
                proj_matrix = self.lsh_proj_matrices[table_idx]
                projections = torch.matmul(vector, proj_matrix.t())
                hash_code = (projections > 0).int()

                # Convert bit array to integer
                hash_value = 0
                for j in range(self.lsh_hash_size):
                    hash_value |= hash_code[j].item() << j # type: ignore [index, operator]

                # Get all centroids in this bucket
                centroid_indices = torch.nonzero(
                    self.lsh_centroids[table_idx, hash_value], as_tuple=True # type: ignore [index]
                )[0]
                candidates_for_vector.update(centroid_indices.tolist())

            candidate_centroids.add(tuple(sorted(candidates_for_vector)))

        # For unique sets of candidates, find nearest centroid
        candidate_sets = list(candidate_centroids)
        nearest_centroids = torch.zeros(batch_size * seq_len, dtype=torch.long)

        for candidate_set in candidate_sets:
            # Find which vectors have this candidate set
            mask = torch.zeros(batch_size * seq_len, dtype=torch.bool)
            # Simplified - in practice, need efficient way to match vectors to candidate sets

            # Get subset of vectors with this candidate set
            vectors_subset = x_flat[mask]

            # Only consider candidate centroids
            candidate_centroids_tensor = self.input_centroids[list(candidate_set)] # type: ignore [index]

            # Compute distances and find nearest
            distances = torch.cdist(vectors_subset, candidate_centroids_tensor)
            min_indices = torch.argmin(distances, dim=1)

            # Map back to original centroid indices
            nearest_centroids[mask] = torch.tensor(
                [list(candidate_set)[i] for i in min_indices]
            )

        return nearest_centroids.reshape(batch_size, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using lookup-based computation

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Find nearest centroid for each input vector
        nearest_centroids = self._find_nearest_centroids(x)

        # First lookup (equivalent to first linear layer)
        hidden_vecs = torch.zeros(
            batch_size, seq_len, self.hidden_dim, dtype=self.dtype
        )
        for b in range(batch_size):
            for s in range(seq_len):
                centroid_idx = nearest_centroids[b, s].item()
                hidden_vecs[b, s] = self.lookup_table_fc1[centroid_idx] # type: ignore [index]

        # Add bias and apply activation
        hidden_vecs = hidden_vecs + self.fc1_bias
        hidden_vecs = self.activation(hidden_vecs)

        # Second lookup (equivalent to second linear layer)
        # For simplicity, we're using the same centroids for both lookups
        # In practice, you might want to have separate centroids for each layer
        output_vecs = torch.zeros(
            batch_size, seq_len, self.output_dim, dtype=self.dtype
        )
        for b in range(batch_size):
            for s in range(seq_len):
                centroid_idx = nearest_centroids[b, s].item()
                output_vecs[b, s] = self.lookup_table_fc2[centroid_idx] # type: ignore [index]

        # Add bias
        output_vecs = output_vecs + self.fc2_bias

        return output_vecs

    @classmethod
    def from_standard_ffn(cls, ffn_module, num_centroids: int = 1024):
        """
        Convert a standard FFN module to LookupFFN

        Args:
            ffn_module: Standard feed-forward module with fc1 and fc2
            num_centroids: Number of centroids for vector quantization

        Returns:
            LookupFFN instance
        """
        # Extract dimensions from the standard FFN module
        input_dim = ffn_module.fc1.in_features
        hidden_dim = ffn_module.fc1.out_features
        output_dim = ffn_module.fc2.out_features

        # Identify activation function
        activation = "gelu"  # default
        for name, module in ffn_module.named_modules():
            if isinstance(module, nn.GELU):
                activation = "gelu"
                break
            elif isinstance(module, nn.ReLU):
                activation = "relu"
                break
            elif isinstance(module, nn.SiLU):
                activation = "silu"
                break

        # Create LookupFFN instance
        lookup_ffn = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_centroids=num_centroids,
            activation=activation,
            dtype=ffn_module.fc1.weight.dtype,
        )

        # Initialize lookup tables based on original weights
        # This would require more sophisticated quantization in practice
        # Here we're just doing a simple K-means-like initialization

        # For first layer
        fc1_weight = ffn_module.fc1.weight.detach()
        # Create pseudo-inputs to train the lookup table
        pseudo_inputs = torch.randn(10000, input_dim, dtype=fc1_weight.dtype)
        # Forward through original layer
        original_outputs = (
            torch.matmul(pseudo_inputs, fc1_weight.t()) + ffn_module.fc1.bias
        )

        # Create lookup table
        for c in range(num_centroids):
            # Simple approximation - in practice needs proper clustering
            lookup_ffn.lookup_table_fc1[c] = original_outputs[c % 10000] # type: ignore [index, operator]

        # Similar approach for second layer
        fc2_weight = ffn_module.fc2.weight.detach()
        pseudo_hidden = torch.randn(10000, hidden_dim, dtype=fc2_weight.dtype)
        original_outputs2 = (
            torch.matmul(pseudo_hidden, fc2_weight.t()) + ffn_module.fc2.bias
        )

        for c in range(num_centroids):
            lookup_ffn.lookup_table_fc2[c] = original_outputs2[c % 10000] # type: ignore [index, operator]

        # Copy biases
        lookup_ffn.fc1_bias.data.copy_(ffn_module.fc1.bias.data)
        lookup_ffn.fc2_bias.data.copy_(ffn_module.fc2.bias.data)

        return lookup_ffn
