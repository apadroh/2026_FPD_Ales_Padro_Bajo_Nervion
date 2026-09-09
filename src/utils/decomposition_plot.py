"""Interactive STL decomposition (Plotly) for hourly series per station."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL


def station_series(
    df: pd.DataFrame,
    station_name: str,
    target: str,
    last_days: int | None = 730,
) -> pd.Series:
    series = (
        df.loc[df["station_name"] == station_name, ["time", target]]
        .dropna()
        .sort_values("time")
        .set_index("time")[target]
    )
    if last_days is not None:
        cutoff = series.index.max() - pd.Timedelta(days=last_days)
        series = series.loc[series.index >= cutoff]
    return series.asfreq("h").interpolate(limit=3).dropna()


def stl_decompose(series: pd.Series, period: int = 24):
    if len(series) < period * 2:
        raise ValueError(
            f"Series too short for period={period} "
            f"({len(series)} observaciones)"
        )
    return STL(series, period=period, robust=True).fit()


def build_stl_figure(
    decompositions: dict[str, object],
    period: int = 24,
    title_prefix: str = "Descomposicion STL de PM10",
) -> go.Figure:
    stations = list(decompositions.keys())
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Observado",
            "Tendencia",
            f"Seasonality ({period} h)",
            "Residuo",
        ],
    )

    component_getters = [
        lambda r: r.observed,
        lambda r: r.trend,
        lambda r: r.seasonal,
        lambda r: r.resid,
    ]

    for i, station in enumerate(stations):
        result = decompositions[station]
        visible = i == 0
        for row, getter in enumerate(component_getters, start=1):
            values = getter(result)
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values,
                    mode="lines",
                    line={"width": 1.1},
                    visible=visible,
                    name=station,
                    showlegend=False,
                ),
                row=row,
                col=1,
            )

    traces_per_station = 4
    buttons = []
    for i, station in enumerate(stations):
        visibility = [False] * len(stations) * traces_per_station
        for j in range(traces_per_station):
            visibility[i * traces_per_station + j] = True
        buttons.append(
            {
                "label": station,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"{title_prefix} — {station}"},
                ],
            }
        )

    fig.update_layout(
        title=f"{title_prefix} — {stations[0]}",
        height=900,
        hovermode="x unified",
        updatemenus=[
            {
                "active": 0,
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        annotations=[
            {
                "text": "Station:",
                "x": 0.0,
                "xref": "paper",
                "y": 1.16,
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 12},
            }
        ],
    )
    fig.update_xaxes(title_text="Tiempo", row=4, col=1)
    fig.update_yaxes(title_text="PM10", row=1, col=1)
    return fig


def decompose_all_stations(
    df: pd.DataFrame,
    target: str = "PM10",
    period: int = 24,
    last_days: int | None = 730,
) -> dict[str, object]:
    decompositions = {}
    for station_name in sorted(df["station_name"].unique()):
        print(f"  STL: {station_name}")
        series = station_series(df, station_name, target, last_days=last_days)
        decompositions[station_name] = stl_decompose(series, period=period)
    return decompositions


def save_stl_html(
    df: pd.DataFrame,
    output_path: str | Path,
    target: str = "PM10",
    period: int = 24,
    last_days: int | None = 730,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decompositions = decompose_all_stations(
        df, target=target, period=period, last_days=last_days
    )
    fig = build_stl_figure(decompositions, period=period, title_prefix=f"STL de {target}")
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path
