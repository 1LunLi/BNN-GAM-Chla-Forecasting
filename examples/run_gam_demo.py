from pathlib import Path

from src.gam_attribution import run_gam_analysis


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "sample_gam_input.xlsx"
OUTPUT_DIR = ROOT_DIR / "outputs" / "gam_demo"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_gam_analysis(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR,
        response_column="Chla_t",
        sheet_name="F1L7",
        n_splines=12,
        lam=0.6,
        grid_points=500,
        dpi=600,
    )

    print(f"GAM analysis completed: {len(results)} result rows")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
