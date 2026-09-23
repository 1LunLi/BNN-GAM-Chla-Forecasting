"""GAM post-hoc interpretation for the AT=1, LT=7 BNN scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pygam import LinearGAM, s


FACTORS = [
    {
        "name": "sin_month",
        "columns": ("sin(month)_avg",),
        "log10": False,
        "label": r"$\sin(\mathrm{month})$",
    },
    {
        "name": "TP",
        "columns": ("TP_avg",),
        "log10": True,
        "label": r"$\log_{10}$[TP (mg/L)]",
    },
    {
        "name": "T",
        "columns": ("T_avg",),
        "log10": False,
        "label": r"$T$ ($^\circ$C)",
    },
    {
        "name": "PAR",
        "columns": ("PAR_avg",),
        "log10": True,
        "label": r"$\log_{10}$[PAR (MJ/m$^2$/day)]",
    },
    {
        "name": "tau_b",
        "columns": (
            "tau_b_avg",
            "bed_shear_stress_avg",
            "\u03c4_avg",
            "\ufffd\ufffd_avg",
        ),
        "log10": True,
        "label": r"$\log_{10}(\tau_b)$ (N/m$^2$)",
    },
    {
        "name": "HR7",
        "columns": ("HR7_avg",),
        "log10": False,
        "label": r"$HR7$ (m)",
    },
]


def find_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first matching column name."""
    for column in candidates:
        if column in data.columns:
            return column

    raise KeyError(
        f"None of these columns were found: {', '.join(candidates)}"
    )


def transform_values(
    values: np.ndarray,
    use_log10: bool,
) -> np.ndarray:
    """Apply the fixed transformation used in the manuscript."""
    if use_log10:
        return np.log10(values)

    return values


def run_gam_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    response_column: str = "Chla_t",
    sheet_name: str = "F1L7",
    n_splines: int = 12,
    lam: float = 0.6,
    grid_points: int = 500,
    dpi: int = 600,
) -> pd.DataFrame:
    """
    Fit univariate GAMs between environmental predictors and BNN output.

    The response column is the BNN-predicted Chl-a concentration for AT=1,
    LT=7. The original Chla_avg column is intentionally excluded.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(input_path, sheet_name=sheet_name)

    if response_column not in data.columns:
        raise KeyError(
            f"Response column not found: {response_column}"
        )

    predicted_chla = pd.to_numeric(
        data[response_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    figure, axes = plt.subplots(
        1,
        len(FACTORS),
        figsize=(18, 3.6),
        squeeze=False,
    )

    curve_tables = []

    for column_index, factor in enumerate(FACTORS):
        source_column = find_column(data, factor["columns"])

        raw_x = pd.to_numeric(
            data[source_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(raw_x)
            & np.isfinite(predicted_chla)
            & (predicted_chla > 0)
        )

        if factor["log10"]:
            valid &= raw_x > 0

        x = transform_values(raw_x[valid], factor["log10"])
        y = np.log10(predicted_chla[valid])

        if len(x) < n_splines + 1:
            raise ValueError(
                f"Insufficient valid observations for {factor['name']}."
            )

        if len(np.unique(x)) < 4:
            raise ValueError(
                f"Too few unique values for {factor['name']}."
            )

        gam = LinearGAM(
            s(0, n_splines=n_splines),
            lam=lam,
        ).fit(x.reshape(-1, 1), y)

        x_grid = np.linspace(x.min(), x.max(), grid_points)
        confidence = gam.confidence_intervals(
            x_grid.reshape(-1, 1),
            width=0.95,
        )
        fitted = gam.predict(x_grid.reshape(-1, 1))

        original_x_grid = (
            10**x_grid
            if factor["log10"]
            else x_grid
        )

        curve_tables.append(
            pd.DataFrame(
                {
                    "scenario": "BNN_AT1LT7",
                    "ahead_time": 1,
                    "lagged_time": 7,
                    "factor": factor["name"],
                    "x_original": original_x_grid,
                    "x_transformed": x_grid,
                    "predicted_log10_chla": fitted,
                    "lower_95": confidence[:, 0],
                    "upper_95": confidence[:, 1],
                }
            )
        )

        axis = axes[0, column_index]

        axis.scatter(
            x,
            y,
            s=10,
            color="#1478B8",
            alpha=0.55,
            edgecolors="none",
        )

        axis.fill_between(
            x_grid,
            confidence[:, 0],
            confidence[:, 1],
            color="#BDBDBD",
            alpha=0.65,
        )

        axis.plot(
            x_grid,
            fitted,
            color="black",
            linewidth=1.6,
        )

        axis.set_xlabel(factor["label"], fontsize=9)
        axis.set_title(factor["name"], fontsize=10)
        axis.grid(
            color="#D9D9D9",
            linewidth=0.6,
            alpha=0.8,
        )
        axis.tick_params(labelsize=8)

    figure.supylabel(
        r"BNN-predicted $\log_{10}$[Chl-a ($\mu$g/L)]",
        fontsize=10,
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="#1478B8",
            markersize=4,
            label="Samples",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.6,
            label="GAM smooth function",
        ),
        Patch(
            facecolor="#BDBDBD",
            alpha=0.65,
            label="95% confidence interval",
        ),
    ]

    figure.legend(
        handles=legend_items,
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    figure.tight_layout(rect=(0.04, 0.03, 1.0, 0.88))

    figure.savefig(
        output_dir / "gam_AT1LT7.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    curves = pd.concat(curve_tables, ignore_index=True)
    curves.to_csv(
        output_dir / "gam_curves_AT1LT7.csv",
        index=False,
    )

    return curves


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit GAMs to BNN-predicted Chl-a for AT=1 and LT=7."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/gam_AT1LT7")
    args = parser.parse_args()

    run_gam_analysis(
        input_path=args.input,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
