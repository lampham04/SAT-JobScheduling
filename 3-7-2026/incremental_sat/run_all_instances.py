import subprocess
import sys
import time
import pandas as pd
from pathlib import Path

# ==============================
# Paths & constants
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / ".." / "datasets" / "30-S"
SOLUTION_DIR = BASE_DIR / ".." / "solutions" / "30-S"

INSTANCE_NAME_FILE = BASE_DIR / ".." / "filenames" / "30-S.txt"
RESULT_CSV = BASE_DIR / ".." / "results" / "results_Lmax_incremental_SAT.csv"

TIMEOUT = 600  # seconds

def main():

    filenames = Path(INSTANCE_NAME_FILE).read_text().splitlines()

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
        sol_file = SOLUTION_DIR / f"{fname}.txt"

        # reset UB
        sol_file.write_text("")

        start = time.time()

        proc = subprocess.Popen(
            [
                sys.executable,
                str(BASE_DIR / "run_one_instance.py"),
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

            if first.startswith("Lmax"):
                our_ub = int(first.split("=")[1].strip())
            elif first == "UNSAT":
                status = first
        except Exception:
            status = "ERROR"


        row = {
            "filename": fname,
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
