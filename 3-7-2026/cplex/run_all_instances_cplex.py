import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
from cplex_Lmax import read_filenames

# ==============================
# Paths & constants
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / ".." / "datasets" / "xx_05_025_125_50_1-L"
SOLUTION_DIR   = BASE_DIR / ".." / "solutions_cplex" / "xx_05_025_125_50_1-L"

XLS_PATH = BASE_DIR / ".." / "data" / "results" /"Results30-40-50.xls"
SHEET_NAME = "50 jobs"
INSTANCE_TYPE = "S"
RESULT_CSV = BASE_DIR / ".." / "results" / "results_xx_05_025_125_50_1_cplex.csv"

TIMEOUT = 601  # seconds

def main():

    #df_filtered, filenames = read_filenames(XLS_PATH, SHEET_NAME, INSTANCE_TYPE)
    filenames = ["10_05_025_125_50_1.GSP", "20_05_025_125_50_1.GSP", "30_05_025_125_50_1.GSP", "40_05_025_125_50_1.GSP", "50_05_025_125_50_1.GSP"]

    # Resume support
    if RESULT_CSV.exists():
        df_done = pd.read_csv(RESULT_CSV)
        done_files = set([]) #set(df_done["filename"])
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
        sol_file = SOLUTION_DIR / f"{fname}.txt"

        # reset UB
        sol_file.write_text("")

        start = time.time()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(BASE_DIR / "run_one_instance_cplex.py"),
                str(instance_path),
                str(sol_file)
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
            lines = sol_file.read_text().splitlines()

            first = lines[0].strip()
            print(first)

            if first.startswith("Lmax"):
                our_ub = first.split("=")[1].strip()
                print(our_ub)
            elif first == "UNSAT":
                status = first
        except Exception:
            status = "ERROR"


        row = {
            "filename": fname,
            #"OPT VALUE": opt_dict.get(fname),
            "OPT VALUE": our_ub,
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
