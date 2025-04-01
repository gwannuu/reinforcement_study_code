import numpy as np
import plotly.graph_objects as go
from collections import Counter


class rank_based_simulate:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.total = 0.0
        for i in range(1, size + 1):
            self.buffer.append(1 / i)
            self.total += 1 / i
        self.probs = [p / self.total for p in self.buffer]
        self.cum_probs = np.cumsum(self.probs)


def plot_pdf_segments_plotly_with_grouping(buffer_size, num_segments, x_range=None):
    per = rank_based_simulate(buffer_size)
    ranks = np.arange(1, buffer_size + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=per.probs,
            mode="lines+markers",
            name="Probability (PDF)",
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
            y1=max(per.probs),
            line=dict(color="red", dash="dash"),
        )
        annotation_text = f"{count}"
        fig.add_annotation(
            x=b,
            y=max(per.probs),
            text=annotation_text,
            showarrow=True,
            arrowhead=1,
            ax=20,
            ay=-20,
        )

    fig.update_layout(
        title=f"Interactive PDF with {num_segments} Segments (Buffer Size = {buffer_size})",
        xaxis_title="Rank (i)",
        yaxis_title="Probability",
        hovermode="closest",
    )

    if x_range is not None:
        fig.update_xaxes(range=x_range)

    fig.show()


plot_pdf_segments_plotly_with_grouping(1000, 15)
