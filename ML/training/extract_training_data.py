"""
Step 1 of 2 — Extract training features from the ISRA CSV package.
=============================================================================
Pulls the 4 blob-ready (calibration-free) features for every physical defect
in the 1238-1800 file range, grouped into the 18-class consolidated taxonomy
from the earlier taxonomy-consolidation work, and saves a training-ready CSV.

Run this BEFORE train_blob_model.py.

REQUIREMENTS
    pip install duckdb pandas

USAGE
    python extract_training_data.py
    (edit ISRA_DATA_DIR below to point at your extracted ISRA CSV package)
"""

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURE
# ---------------------------------------------------------------------------
ISRA_DATA_DIR = "clean_package"   # folder containing Coils.csv, Defect classes.csv, "Defect data/" etc.
SAMPLES_PER_GROUP_CAP = 30000     # cap per group so huge classes (Bright/Dark Spot) don't dominate training time
OUTPUT_CSV = "training_data.csv"

DEFECT_FILES = [
    "Defect data/1238-1300.csv",
    "Defect data/1301-1400.csv",
    "Defect data/1401-1500 2.csv",
    "Defect data/1501-1600.csv",
    "Defect data/1601-1700.csv",
    "Defect data/1701-1800.csv",
]

# The 18-class consolidated taxonomy (from the taxonomy-consolidation pass):
# tiered families collapsed to one label each, rare/ungoverned classes merged
# where isolated testing confirmed genuine overlap, everything else kept as
# its own class. See ISRA_Taxonomy_Consolidation.html for the full evidence
# behind every grouping decision.
GROUP_SQL = """
CASE
  WHEN d.Class IN (30,31,32,33,39) THEN 'Bright Spot (family)'
  WHEN d.Class IN (50,51,52,53,59) THEN 'Dark Spot (family)'
  WHEN d.Class IN (21,22,23,29) THEN 'Hole (family)'
  WHEN d.Class IN (80,81,82) THEN 'Streak/Scratch (family)'
  WHEN d.Class IN (60,62) THEN 'Edge Crack (family)'
  WHEN d.Class IN (200,180) THEN 'Minor Surface Anomaly'
  ELSE dc.Name
END
"""

# Excluded at the Stage 0 data-cleaning pass:
#   90  = Streak Short (data anomaly — impossible size values on every row)
#   100 = Formation Irregularity (system/report event, not a physical defect)
#   3   = Unknown (catch-all label, not a real class)
#   0, 2 = Not Classified / Pseudo (system codes)
EXCLUDED_CLASSES = "0,2,90,100,3"


def extract_training_data(data_dir: str = ISRA_DATA_DIR, output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
    con = duckdb.connect()
    files = [f"{data_dir}/{f}" for f in DEFECT_FILES]
    defects_sql = f"read_csv_auto({files}, union_by_name=True)"

    query = f"""
    WITH d0 AS (
      SELECT
        d.Class,
        ({GROUP_SQL.replace('d.', 'd.')}) AS Stage1aGroup,
        d.SizeCD, d.SizeMD, d.A, d.PERC_BRIGHT, d.PERC_DARK,
        (d.RoiX1 - d.RoiX0) AS roi_w,
        (d.RoiY1 - d.RoiY0) AS roi_h
      FROM {defects_sql} d
      LEFT JOIN read_csv_auto('{data_dir}/Defect classes.csv') dc ON d.Class = dc.ClassId
      WHERE d.MergedTo < 0                                   -- logical objects only (Category 4 finding)
        AND d.Class < 1000 AND d.Class NOT IN ({EXCLUDED_CLASSES})
        AND d.SizeCD > 0 AND d.SizeMD > 0
        AND d.SizeCD < 1000 AND d.SizeMD < 1000               -- Stage 0: drop known outlier rows
        AND (d.RoiX1 - d.RoiX0) > 0 AND (d.RoiY1 - d.RoiY0) > 0
    ),
    d1 AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY Stage1aGroup ORDER BY random()) AS rn
      FROM d0
    )
    SELECT Class, Stage1aGroup, SizeCD, SizeMD, A, PERC_BRIGHT, PERC_DARK, roi_w, roi_h
    FROM d1
    WHERE rn <= {SAMPLES_PER_GROUP_CAP}
    """
    df = con.execute(query).df()

    # Derive the 4 blob-ready (calibration-free) features actually used by the model
    df["aspect_ratio"] = df["SizeMD"] / df["SizeCD"]
    df["roi_area"] = df["roi_w"] * df["roi_h"]
    df["A_to_roi_ratio"] = df["A"] / df["roi_area"]

    df.to_csv(output_csv, index=False)

    print(f"Extracted {len(df)} rows across {df['Stage1aGroup'].nunique()} groups")
    print(df["Stage1aGroup"].value_counts().to_string())
    print(f"\nSaved to {output_csv}")

    return df


if __name__ == "__main__":
    extract_training_data()
