import subprocess
import sys
import time
import pandas as pd
from pathlib import Path

# ==============================
# Paths & constants
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets" / "S"
UB_DIR   = BASE_DIR / "datasets" / "UB"

XLS_PATH = BASE_DIR / "datasets" / "Results30-40-50.xls"
RESULT_CSV = BASE_DIR / "results_incremental_sat_600s.csv"

TIMEOUT = 600  # seconds

# ==============================
# Helper: read filenames + OPT
# ==============================

def read_filenames(xls_path):
    df = pd.read_excel(xls_path, engine="xlrd")

    # normalize column names if needed
    df.columns = [c.strip() for c in df.columns]

    # keep valid OPT values
    df["OPT VALUE"] = pd.to_numeric(df["OPT VALUE"], errors="coerce")

    df = df[df["OPT VALUE"].notna()]
    df = df[df["OPT VALUE"] <= 1000]

    filenames = df["filename"].astype(str).tolist()
    return df, filenames


# ==============================
# Main experiment
# ==============================

def main():

    df_filtered, filenames = read_filenames(XLS_PATH)
    opt_dict = dict(zip(df_filtered["filename"], df_filtered["OPT VALUE"]))

    # Resume support
    if RESULT_CSV.exists():
        df_done = pd.read_csv(RESULT_CSV)
        done_files = set(df_done["filename"])
        csv_exists = True
        print(f"🔁 Resume mode: {len(done_files)} instances already done")
    else:
        done_files = set()
        csv_exists = False

    for fname in filenames:

        if fname in done_files:
            print(f"⏩ Skip {fname} (already done)")
            continue

        print(f"\n▶ Running {fname}")

        instance_path = DATA_DIR / fname
        ub_file = UB_DIR / f"{fname}.ub"

        # reset UB
        ub_file.write_text("inf")

        start = time.time()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(BASE_DIR / "run_one_instance.py"),
                str(instance_path),
                str(ub_file)
            ]
        )

        try:
            proc.wait(timeout=TIMEOUT)
            status = "FINISHED"
        except subprocess.TimeoutExpired:
            proc.kill()
            status = "TIMEOUT"

        elapsed = time.time() - start

        # read UB
        our_ub = None
        try:
            text = ub_file.read_text().strip()
            if text.isdigit():
                our_ub = int(text)
            else:
                status = text  # UNSAT / INVALID / ERROR
        except:
            status = "ERROR"

        row = {
            "filename": fname,
            "OPT VALUE": opt_dict.get(fname),
            "OUR OPT VALUE": our_ub,
            "STATUS": status,
            "TIME (s)": round(elapsed, 2)
        }

        pd.DataFrame([row]).to_csv(
            RESULT_CSV,
            mode="a",
            header=not csv_exists,
            index=False
        )

        csv_exists = True
        done_files.add(fname)

        print(f"  ✔ {status} | UB = {our_ub}")

    print("\n✅ Experiment finished (resume-safe)")


if __name__ == "__main__":
    main()
