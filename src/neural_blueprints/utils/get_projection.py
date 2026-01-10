from ..components.composite.projections.input import BaseInputProjection
from ..components.composite.projections.output import BaseOutputProjection

def get_input_projection(projection_config: object) -> BaseInputProjection:
    """Retrieve the input projection class based on the projection config object.

    Args:
        projection_config (callable): The configuration object of the output projection.

    Returns:
        class: The input projection class.
    """
    if projection_config.__class__.__name__ == "TabularInputProjectionConfig":
        from ..components.composite.projections.input.tabular import TabularInputProjection
        return TabularInputProjection(config=projection_config)
    elif projection_config.__class__.__name__ == "LinearInputProjectionConfig":
        from ..components.composite.projections.input.linear import LinearInputProjection
        return LinearInputProjection(config=projection_config)
    elif projection_config.__class__.__name__ == "MultiModalInputProjectionConfig":
        from ..components.composite.projections.input.multimodal import MultiModalInputProjection
        return MultiModalInputProjection(config=projection_config)
    else:
        raise ValueError(f"Unsupported input projection type: {projection_config.__class__.__name__}")

def get_output_projection(projection_config: object) -> BaseOutputProjection:
    """Retrieve the output projection class based on the projection config object.

    Args:
        projection_config (callable): The configuration object of the output projection.

    Returns:
        class: The output projection class.
    """
    if projection_config.__class__.__name__ == "TabularOutputProjectionConfig":
        from ..components.composite.projections.output import TabularOutputProjection
        return TabularOutputProjection(config=projection_config)
    elif projection_config.__class__.__name__ == "LinearOutputProjectionConfig":
        from ..components.composite.projections.output import LinearOutputProjection
        return LinearOutputProjection(config=projection_config)
    else:
        raise ValueError(f"Unsupported output projection type: {projection_config.__class__.__name__}")