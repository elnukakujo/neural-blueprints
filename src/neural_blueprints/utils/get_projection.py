def get_input_projection(projection_config: object) -> object:
    """Retrieve the input projection class based on the projection config object.

    Args:
        projection_config (callable): The configuration object of the output projection.

    Returns:
        class: The input projection class.
    """
    if projection_config.__class__.__name__ == "TabularInputProjectionConfig":
        from ..components.composite.projections.input_projections.tabular import TabularInputProjection
        return TabularInputProjection(config=projection_config)
    else:
        raise ValueError(f"Unsupported input projection type: {projection_config.__class__.__name__}")

def get_output_projection(projection_config: object) -> object:
    """Retrieve the output projection class based on the projection config object.

    Args:
        projection_config (callable): The configuration object of the output projection.

    Returns:
        class: The output projection class.
    """
    if projection_config.__class__.__name__ == "TabularOutputProjectionConfig":
        from ..components.composite.projections.output_projections.tabular import TabularOutputProjection
        return TabularOutputProjection(config=projection_config)
    else:
        raise ValueError(f"Unsupported output projection type: {projection_config.__class__.__name__}")