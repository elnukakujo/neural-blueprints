import torch

from . import (
    BaseInputProjection,
    LinearInputProjection,
    TabularInputProjection
)

from .....config.components.composite.projections.input import (
    LinearInputProjectionConfig,
    TabularInputProjectionConfig,
    MultiModalInputProjectionConfig
)

class MultiModalInputProjection(BaseInputProjection):
    """
    Multi-modal input projection.
    """
    def __init__(
            self,
            config: MultiModalInputProjectionConfig
        ):
        super().__init__()
        if config.tabular_cardinalities is not None:
            self.tabular_projection = TabularInputProjection(
                TabularInputProjectionConfig(
                    cardinalities=config.tabular_cardinalities,
                    latent_dim=config.latent_dim,
                    hidden_dims=config.hidden_dims,
                    activation=config.activation,
                    normalization=config.normalization,
                    dropout_p=config.dropout_p
                )
            )
        else:
            self.tabular_projection = None

        if config.representation_input_dim is not None:
            self.representation_projection = LinearInputProjection(
                LinearInputProjectionConfig(
                    input_dim=config.representation_input_dim,
                    latent_dim=config.latent_dim,
                    hidden_dims=config.hidden_dims,
                    activation=config.activation,
                    normalization=config.normalization,
                    dropout_p=config.dropout_p
                )
            )
        else:
            self.representation_projection = None

        self.input_dim = config.tabular_cardinalities if config.tabular_cardinalities is not None else []
        self.input_dim += config.representation_input_dim if config.representation_input_dim is not None else []
        self.input_dim = tuple(self.input_dim)
        print(self.input_dim)
        self.output_dim = [len(self.input_dim), config.latent_dim]

    
    def forward(self, inputs: dict[dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the multi-modal input projection.

        Args:
            inputs (dict): Dictionary containing 'tabular' and 'representation' tensors.
        
        Returns:
            torch.Tensor: Concatenated projected tensor of shape (batch_size, num_elements, latent_dim).
        """
        if type(inputs) == torch.Tensor:    # For torchinfo forward pass scenario
            tabular_inputs = inputs[:, 0]
            representation_inputs = inputs
        tabular_inputs = inputs['tabular']                  # shape (batch_size, num_tabular_attributes)
        representation_inputs = inputs['representation']    # shape (batch_size, representation_input_dim)

        embeddings = []
        nan_masks = []
        if self.tabular_projection is not None:
            if isinstance(tabular_inputs, torch.Tensor):
                tabular_inputs = {'tabular_input': tabular_inputs}

            for inputs in tabular_inputs.values():
                embedding, mask = self.tabular_projection(inputs)
                embeddings.append(embedding)  # shape: (batch_size, num_attributes, latent_dim)
                nan_masks.append(mask)        # shape: (batch_size, num_attributes)

        if self.representation_projection is not None:
            if isinstance(representation_inputs, torch.Tensor):
                representation_inputs = {'representation_input': representation_inputs}
            
            for inputs in representation_inputs.values():
                inputs = torch.flatten(inputs, start_dim=1)
                embedding = self.representation_projection(inputs)  # shape: (batch_size, num_attributes, latent_dim)
                if len(embedding.size()) == 2:
                    embedding = embedding.unsqueeze(1)  # shape: (batch_size, 1, latent_dim)
                embeddings.append(embedding)  # shape: (batch_size, num_attributes, latent_dim)

                batch_size = embedding.size(0)
                rep_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=embedding.device)
                nan_masks.append(rep_mask)
        
        assert len(embeddings) > 0, "At least one modality must be provided."

        concatenated_embeddings = torch.cat(embeddings, dim=1)  # shape: (batch_size, total_num_elements, latent_dim)
        nan_masks = torch.cat(nan_masks, dim=1)  # shape: (batch_size, total_num_elements)

        return concatenated_embeddings, nan_masks