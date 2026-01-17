from ..components.composite.projections import BaseProjection

def get_projection(projection_config: object) -> BaseProjection:
    if projection_config.__class__.__name__ == "TabularProjectionConfig":
        from ..components.composite.projections import TabularProjection
        return TabularProjection(config=projection_config)
    elif projection_config.__class__.__name__ == "LinearProjectionConfig":
        from ..components.composite.projections import LinearProjection
        return LinearProjection(config=projection_config)
    elif projection_config.__class__.__name__ == "MultiModalProjectionConfig":
        from ..components.composite.projections.multimodal import MultiModalInputProjection
        return MultiModalInputProjection(config=projection_config)
    else:
        raise ValueError(f"Unsupported input projection type: {projection_config.__class__.__name__}")