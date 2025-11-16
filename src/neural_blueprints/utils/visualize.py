import math
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def curve_plot(
    x,
    ys,
    labels=None,
    title="Curve Plot",
    subtitle=None,
    x_label="X-axis",
    y_label="Y-axis",
    show_min_max=True,
    colors=None,
    markers=True
):
    """
    Modern, clean curve plot with optional multiple curves and annotations.
    
    Parameters:
    - x: list or array of x values
    - ys: list of y arrays/lists. If single curve, can be a single list
    - labels: list of labels for each curve
    - title: main title
    - subtitle: optional subtitle
    - x_label, y_label: axis labels
    - show_min_max: if True, annotate min and max points
    - colors: list of colors for curves
    - markers: if True, show markers at data points
    """
    if not isinstance(ys, list) or isinstance(ys[0], (int, float)):
        ys = [ys]  # single curve to list
    
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(ys))]
    
    if colors is None:
        colors = [None] * len(ys)
    
    fig = go.Figure()
    
    for i, y in enumerate(ys):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers' if markers else 'lines',
                name=labels[i],
                line=dict(width=3, color=colors[i]),
                marker=dict(size=6) if markers else None,
                hovertemplate=f"%{{x}}, %{{y}}<extra>{labels[i]}</extra>"
            )
        )
        
        # Annotate min and max points
        if show_min_max:
            min_idx = y.index(min(y)) if isinstance(y, list) else y.argmin()
            max_idx = y.index(max(y)) if isinstance(y, list) else y.argmax()
            
            fig.add_annotation(
                x=x[min_idx], y=y[min_idx],
                text=f"Min: {y[min_idx]:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=-20, ay=-40,
                font=dict(color="red")
            )
            fig.add_annotation(
                x=x[max_idx], y=y[max_idx],
                text=f"Max: {y[max_idx]:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-40,
                font=dict(color="green")
            )
    
    # Layout styling
    fig.update_layout(
        title=dict(
            text=title + ("<br><sup>" + subtitle + "</sup>" if subtitle else ""),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=14),
        legend=dict(title="Curves", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.show()

def bar_plot(categories, values, title="Bar Plot", x_label="Categories", y_label="Values"):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()

def heatmap_plot(z, x=None, y=None, title="Heatmap", x_label="X-axis", y_label="Y-axis"):
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()

def image_plot(images, title="Images", cols=4):
    """
    Display images in a grid using matplotlib.
    Handles grayscale (H, W, 1), RGB, torch tensors, and numpy arrays.
    """

    # Convert all images to numpy + squeeze grayscale channel
    processed = []
    for img in images:
        if hasattr(img, "detach"):   # torch tensor
            img = img.detach().cpu().numpy()

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)  # (H, W)

        processed.append(img)

    n = len(processed)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # If only one row/col, wrap axes into a list
    axes = np.array(axes).reshape(rows, cols)

    index = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]

            if index < n:
                img = processed[index]

                # grayscale
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)

            ax.axis("off")
            index += 1

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()