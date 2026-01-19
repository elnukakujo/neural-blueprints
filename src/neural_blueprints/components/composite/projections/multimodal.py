import re
import torch
import torch.nn as nn
import numpy as np

from neural_blueprints.types.sample import MultiModalSample

from . import (
    BaseProjection,
    LinearProjection,
    TabularProjection
)

from ....config.components.composite.projections import (
    LinearProjectionConfig,
    TabularProjectionConfig,
    MultiModalProjectionConfig
)

from ....types import (
    UniModalDim,
    MultiModalDim,
    SingleProjectionDim,
    MultiProjectionDim,
    MultiModalSample,
    UniModalSample
)

def parse_index_spec(spec: str) -> list[int]:
    """
    Parse an index specification string into a list of indices and concatenation flag.
    
    Args:
        spec: Index specification (e.g., "0-3+5*", "4+6", "7")
    
    Returns:
        List of indices
    
    Examples:
        "0-3+5*" -> [0, 1, 2, 3, 5]
        "4+6" -> [4, 6]
        "7" -> [7]
    """
    
    indices = []
    
    # Split by '+' to get individual parts
    parts = spec.split('+')
    
    for part in parts:
        part = part.strip()
        
        # Check if it's a range (e.g., "0-3")
        if '-' in part:
            range_match = re.match(r'^(\d+)-(\d+)$', part)
            if range_match:
                start, end = map(int, range_match.groups())
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                indices.extend(range(start, end + 1))
            else:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Single index
            if not part.isdigit():
                raise ValueError(f"Invalid index: {part}")
            indices.append(int(part))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    
    return unique_indices

def _parse_multi_projection_dim(multi_proj_dim: MultiProjectionDim) -> dict[str, tuple[int]]:
    parsed_projection_dims = {}
    for modality_type, spec in multi_proj_dim.items():
        if isinstance(spec, tuple):
            parsed_projection_dims[modality_type] = spec
        elif isinstance(spec, dict):
            parsed_projection_dims[modality_type] = {}
            group_idx = 0
            for index_spec, dim in spec.items():
                indices = parse_index_spec(index_spec)
                parsed_projection_dims[modality_type][f"group_{group_idx}"] = {}
                for idx in indices:
                    parsed_projection_dims[modality_type][f"group_{group_idx}"][idx] = dim

                group_idx += 1
        else:
            raise ValueError(f"Invalid spec type for modality '{modality_type}': {type(spec)}")
    return parsed_projection_dims

def get_projection_module(config, modality_type: str, modality_dim: tuple[int], projection_dim: tuple[int], direction: str) -> nn.Module:
    if modality_type == "tabular":
        tabular_config = TabularProjectionConfig(
            input_cardinalities=modality_dim if direction == "input" else None,
            output_cardinalities=modality_dim if direction == "output" else None,
            input_dim=projection_dim if direction == "output" else None,
            output_dim=projection_dim if direction == "input" else None,
            **config.dict(exclude={"projection_dim", "input_modalities_dim", "output_modalities_dim"})
        )
        return TabularProjection(tabular_config)
    elif modality_type == "representation":
        linear_config = LinearProjectionConfig(
            input_dim=modality_dim if direction == "input" else projection_dim,
            output_dim=modality_dim if direction == "output" else projection_dim,
            **config.dict(exclude={"projection_dim", "input_modalities_dim", "output_modalities_dim"})
        )
        return LinearProjection(linear_config)
    else:
        raise NotImplementedError(f"Projection for modality type '{modality_type}' is not implemented.")
    

class MultiModalInputProjection(BaseProjection):
    def __init__(self, config: MultiModalProjectionConfig):
        super().__init__()

        modalities_dims = config.input_modalities_dim
        raw_projection_dims = config.projection_dim

        if isinstance(raw_projection_dims, tuple):
            projection_dims = {
                k: raw_projection_dims
                for k in modalities_dims.keys()
            }
        elif isinstance(raw_projection_dims, dict):
            projection_dims = _parse_multi_projection_dim(raw_projection_dims)
        else:
            raise ValueError("Invalid projection_dim type.")

        self.projections = nn.ModuleDict()
        for modality_type, modality_dim in modalities_dims.items():
            if isinstance(modality_dim, tuple):
                assert isinstance(projection_dims[modality_type], tuple), f"Projection dimension for modality '{modality_type}' must match modality dimension type."
                output_dim = projection_dims[modality_type]
                self.projections[modality_type] = get_projection_module(
                    config=config,
                    modality_type=modality_type,
                    modality_dim=modality_dim,
                    projection_dim=output_dim,
                    direction="input"
                )
            elif isinstance(modality_dim, dict):
                assert isinstance(projection_dims[modality_type], dict), f"Projection dimension for modality '{modality_type}' must match modality dimension type."
                self.projections[modality_type] = nn.ModuleDict()
                for group_idx, group_projection_dim in projection_dims[modality_type].items():
                    index = group_projection_dim.keys()

                    input_dims = [list(modality_dim.values())[idx] for idx in index]
                    output_dims = [group_projection_dim[idx] for idx in index]
                    
                    input_dim = input_dims[0]
                    output_dim = output_dims[0]

                    self.projections[modality_type][group_idx] = get_projection_module(
                        config=config,
                        modality_type=modality_type,
                        modality_dim=input_dim,
                        projection_dim=output_dim,
                        direction="input"
                    )
            else:
                raise ValueError(f"Invalid type for modality '{modality_type}'.")
                        
        self.input_dim = modalities_dims
        self.output_dim = projection_dims
        
    def forward(self, x: MultiModalSample) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Forward pass for multi-modal input projection.
        
        Args:
            x: MultiModalSample containing data for each modality
        
        Returns:
            Dictionary mapping modality types to:
            - torch.Tensor for simple projections (no grouping)
            - dict[str, torch.Tensor] for grouped projections (one tensor per group)
        """
        outputs = {}
        
        for modality_type, modality_data in x.items():
            if modality_type not in self.projections:
                raise KeyError(f"Unknown modality type: {modality_type}")
            
            projection = self.projections[modality_type]
            
            if isinstance(projection, nn.ModuleDict):
                # Grouped projections - keep groups separate
                outputs[modality_type] = {}
                
                for group_idx, group_projection in projection.items():
                    # Get the indices for this group from output_dim
                    group_config = self.output_dim[modality_type][group_idx]
                    indices = list(group_config.keys())
                    
                    # Extract features for this group
                    if isinstance(modality_data, dict):
                        # Data is a dictionary of features
                        group_features = [list(modality_data.values())[idx] for idx in indices]
                    elif isinstance(modality_data, (list, tuple)):
                        # Data is a list/tuple of features
                        group_features = [modality_data[idx] for idx in indices]
                    else:
                        # Data is a single tensor with features along a dimension
                        # Assume features are along dimension 1: [batch, num_features, ...]
                        group_features = [modality_data[:, idx] for idx in indices]

                    # Store this group's output separately
                    outputs[modality_type][group_idx] = torch.stack([group_projection(feature) for feature in group_features], dim=1)            
            else:
                # Simple projection (no grouping)
                outputs[modality_type] = projection(modality_data)
        
        return outputs
        
class MultiModalOutputProjection(BaseProjection):
    def __init__(self, config: MultiModalProjectionConfig):
        modalities_dims = config.output_modalities_dim
        raw_projection_dims = config.projection_dim

        if isinstance(raw_projection_dims, tuple):
            projection_dims = {
                k: raw_projection_dims
                for k in modalities_dims.keys()
            }
        elif isinstance(raw_projection_dims, dict):
            projection_dims = _parse_multi_projection_dim(raw_projection_dims)
        else:
            raise ValueError("Invalid projection_dim type.")

        self.projections = nn.ModuleDict()
        for modality_type, modality_dim in modalities_dims.items():
            if isinstance(modality_dim, tuple):
                assert isinstance(projection_dims[modality_type], tuple), f"Projection dimension for modality '{modality_type}' must match modality dimension type."
                output_dim = projection_dims[modality_type]
                self.projections[modality_type] = get_projection_module(
                    config=config,
                    modality_type=modality_type,
                    modality_dim=modality_dim,
                    projection_dim=output_dim,
                    direction="output"
                )
            elif isinstance(modality_dim, dict):
                assert isinstance(projection_dims[modality_type], dict), f"Projection dimension for modality '{modality_type}' must match modality dimension type."
                self.projections[modality_type] = nn.ModuleDict()
                for group_idx, group_projection_dim in projection_dims[modality_type].items():
                    index = group_projection_dim.keys()

                    input_dims = [list(modality_dim.values())[idx] for idx in index]
                    output_dims = [group_projection_dim[idx] for idx in index]
                    
                    input_dim = input_dims[0]
                    output_dim = output_dims[0]

                    self.projections[modality_type][group_idx] = get_projection_module(
                        config=config,
                        modality_type=modality_type,
                        modality_dim=input_dim,
                        projection_dim=output_dim,
                        direction="output"
                    )
            else:
                raise ValueError(f"Invalid type for modality '{modality_type}'.")
                        
        self.input_dim = projection_dims
        self.output_dim = modalities_dims

    def forward(self, x: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> MultiModalSample:
        """
        Forward pass for multi-modal output projection.
        Projects from projection space back to multi-modal space.
        
        Args:
            x: Dictionary mapping modality types to:
            - torch.Tensor for simple projections
            - dict[str, torch.Tensor] for grouped projections (one tensor per group)
        
        Returns:
            MultiModalSample with reconstructed modality data
        """
        outputs = {}
        
        for modality_type, modality_data in x.items():
            if modality_type not in self.projections:
                raise KeyError(f"Unknown modality type: {modality_type}")
            
            projection = self.projections[modality_type]
            
            if isinstance(projection, nn.ModuleDict):
                # Grouped projections - process each group separately
                all_features = []
                
                for group_idx, group_projection in projection.items():
                    # Get input for this group
                    if isinstance(modality_data, dict):
                        group_input = modality_data[group_idx]
                    else:
                        raise ValueError(f"Expected dict input for grouped modality '{modality_type}', got {type(modality_data)}")
                    
                    # Get the configuration for this group
                    group_config = self.input_dim[modality_type][group_idx]
                    indices = list(group_config.keys())
                    num_features = len(indices)
                    
                    # Get the projection dimension for one feature in this group
                    proj_dim = list(group_config.values())[0]
                    
                    # Split the concatenated group input back into individual features
                    # group_input shape: [batch, num_features * proj_dim]
                    batch_size = group_input.shape[0]
                    
                    # Reshape to separate features: [batch, num_features, proj_dim]
                    group_input_reshaped = group_input.view(batch_size, num_features, proj_dim)
                    
                    # Project each feature back to output space
                    group_features = []
                    for i in range(num_features):
                        feature_input = group_input_reshaped[:, i, :]  # [batch, proj_dim]
                        feature_output = group_projection(feature_input)  # [batch, output_dim]
                        group_features.append((indices[i], feature_output))
                    
                    all_features.extend(group_features)
                
                # Sort features by index and create output
                all_features.sort(key=lambda x: x[0])
                
                # Return as dict or list based on output format preference
                outputs[modality_type] = {idx: feat for idx, feat in all_features}
                # Alternative: outputs[modality_type] = [feat for _, feat in all_features]
            
            else:
                # Simple projection (no grouping)
                outputs[modality_type] = projection(modality_data)
        
        return outputs

class MultiModalProjection(BaseProjection):
    def __new__(cls, config: MultiModalProjectionConfig):
        if config.input_modalities_dim:
            return MultiModalInputProjection(config)
        elif config.output_modalities_dim:
            return MultiModalOutputProjection(config)
        else:
            raise ValueError("Either input_modalities_dim or output_modalities_dim must be provided in the config.")