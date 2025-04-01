import numpy as np
import plotly.graph_objects as go
from collections import Counter


class PER_simulate:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.total = 0.0
        for i in range(1, size + 1):
            self.buffer.append(1 / i)
            self.total += 1 / i
        self.probs = np.array(self.buffer) / self.total
        self.cum_probs = np.cumsum(self.probs)


def compute_normalized_weights(P, beta, N):
    raw = (1 / (N * P)) ** beta
    return raw / np.max(raw)


def plot_normalized_weights_plotly_with_grouping(
    buffer_size, num_segments, beta=1.0, x_range=None
):
    per = PER_simulate(buffer_size)
    ranks = np.arange(1, buffer_size + 1)
    N = buffer_size

    norm_weights = compute_normalized_weights(per.probs, beta, N)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks, y=norm_weights, mode="lines+markers", name="Normalized Weights"
        )
    )

    boundaries = []
    for seg in range(1, num_segments):
        target = seg / num_segments
        idx = np.searchsorted(per.cum_probs, target)
        boundaries.append(idx + 1)

    boundary_counts = Counter(boundaries)
    unique_boundaries = sorted(boundary_counts.keys())

    for b in unique_boundaries:
        count = boundary_counts[b]
        fig.add_shape(
            type="line",
            x0=b,
            y0=0,
            x1=b,
            y1=max(norm_weights),
            line=dict(color="red", dash="dash"),
        )
        annotation_text = f"{count}"
        fig.add_annotation(
            x=b,
            y=max(norm_weights),
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-20,
        )

    fig.update_layout(
        title=f"Interactive Normalized Weights (β={beta}) with {num_segments} Segments\n(Buffer Size = {buffer_size})",
        xaxis_title="Rank (i)",
        yaxis_title="Normalized Weight",
        hovermode="closest",
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)

    fig.show()


plot_normalized_weights_plotly_with_grouping(1000, 15, beta=1.0)
