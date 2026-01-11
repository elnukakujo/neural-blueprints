import torch
import torch.nn as nn

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

        input_spec = config.input_spec
        self.input_dim = []

        if input_spec["tabular"] is not None:
            if isinstance(input_spec["tabular"], dict):
                self.tabular_projection = nn.ModuleDict()
                for key, cardinalities in input_spec["tabular"].items():
                    self.tabular_projection[key] = TabularInputProjection(
                        TabularInputProjectionConfig(
                            cardinalities=cardinalities,
                            latent_dim=config.latent_dim,
                            hidden_dims=config.hidden_dims,
                            activation=config.activation,
                            normalization=config.normalization,
                            dropout_p=config.dropout_p
                        )
                    )
                    self.input_dim.append([len(cardinalities)])
            elif isinstance(input_spec["tabular"], tuple):
                self.tabular_projection = TabularInputProjection(
                    TabularInputProjectionConfig(
                        cardinalities=input_spec["tabular"],
                        latent_dim=config.latent_dim,
                        hidden_dims=config.hidden_dims,
                        activation=config.activation,
                        normalization=config.normalization,
                        dropout_p=config.dropout_p
                    )
                )
                self.input_dim.append([len(list(input_spec["tabular"]))])
            else:
                raise ValueError("Invalid type for tabular input_spec.")
        else:
            self.tabular_projection = None

        if input_spec["representation"] is not None:
            if isinstance(input_spec["representation"], dict):
                self.representation_projection = nn.ModuleDict()
                for key, input_dim in input_spec["representation"].items():
                    input_dim = list(input_dim)
                    self.representation_projection[key] = LinearInputProjection(
                        LinearInputProjectionConfig(
                            input_dim=input_dim,
                            latent_dim=config.latent_dim,
                            hidden_dims=config.hidden_dims,
                            activation=config.activation,
                            normalization=config.normalization,
                            dropout_p=config.dropout_p
                        )
                    )
                    self.input_dim.append(input_dim)
            elif isinstance(input_spec["representation"], tuple):
                self.representation_projection = LinearInputProjection(
                    LinearInputProjectionConfig(
                        input_dim=input_spec["representation"],
                        latent_dim=config.latent_dim,
                        hidden_dims=config.hidden_dims,
                        activation=config.activation,
                        normalization=config.normalization,
                        dropout_p=config.dropout_p
                    )
                )
                self.input_dim.append(list(input_spec["representation"]))
            else:
                raise ValueError("Invalid type for representation input_spec.")
        else:
            self.representation_projection = None

        self.input_dim = tuple(self.input_dim)
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

        try:
            tabular_inputs = inputs['tabular']                  # shape (batch_size, num_tabular_attributes)
            representation_inputs = inputs['representation']    # shape (batch_size, representation_input_dim)
        except IndexError:
            print(inputs)
            raise ValueError("Inputs must be a dictionary with 'tabular' and 'representation' keys.")

        embeddings = []
        nan_masks = []
        if self.tabular_projection is not None:
            if isinstance(tabular_inputs, torch.Tensor):
                embedding, mask = self.tabular_projection(tabular_inputs)
                embeddings.append(embedding)  # shape: (batch_size, num_attributes, latent_dim
                nan_masks.append(mask)        # shape: (batch_size, num_attributes)
            elif isinstance(tabular_inputs, dict):
                for inputs, tabular_projection in zip(tabular_inputs.values(), self.tabular_projection.values()):
                    embedding, mask = tabular_projection(inputs)
                    embeddings.append(embedding)  # shape: (batch_size, num_attributes, latent_dim)
                    nan_masks.append(mask)        # shape: (batch_size, num_attributes)
            else:
                raise ValueError("Tabular inputs must be a tensor or a dictionary of tensors.")

        if self.representation_projection is not None:
            if isinstance(representation_inputs, torch.Tensor):
                inputs = torch.flatten(representation_inputs, start_dim=1)
                embedding = self.representation_projection(inputs)  # shape: (batch_size, num_attributes, latent_dim)
                if len(embedding.size()) == 2:
                    embedding = embedding.unsqueeze(1)  # shape: (batch_size, 1, latent_dim)
                embeddings.append(embedding)  # shape: (batch_size, num_attributes, latent_dim)

                batch_size = embedding.size(0)
                rep_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=embedding.device)
                nan_masks.append(rep_mask)
            elif isinstance(representation_inputs, dict):
                for inputs, representation_projection in zip(representation_inputs.values(), self.representation_projection.values()):
                    inputs = torch.flatten(inputs, start_dim=1)
                    embedding = representation_projection(inputs)  # shape: (batch_size, num_attributes, latent_dim)
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