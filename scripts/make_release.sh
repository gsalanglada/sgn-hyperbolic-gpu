#!/bin/bash
# Assemble the reproducibility package (GitHub repository + Zenodo archive).
#
#   bash scripts/make_release.sh [outdir]            # default outdir: sgn-hyperbolic-gpu
#   RESULTS_DIR=/path/to/results bash scripts/make_release.sh dist/sgn-hyperbolic-gpu
#
# Run from the repository root. RESULTS_DIR (default: results) is where the
# reference output lives; on MN5 that is the repository's own results/.
# Produces <outdir>/ plus <outdir>.tar.gz.
#
# Deliberately excluded: build artefacts, every VTK visualisation dump (shoal
# ~180 MB, tsunami ~800 MB; regenerable and not needed for any quantitative
# result), raw Nsight reports (*.nsys-rep, *.sqlite: binary and tool-version
# specific; their text summaries ARE shipped), editor backups (*.bak), the
# authors' working documents and the paper sources.
#
# Steps: stage code (src, apps, test, Makefile) -> scripts -> repository
# metadata -> digitized lab data -> reference results -> build-environment
# record -> tar.gz. Only files named explicitly below are shipped:
# a new script or document must be added to the copy lists to be included.

set -euo pipefail

OUT=${1:-sgn-hyperbolic-gpu}
RES=${RESULTS_DIR:-results}
ROOT=$(pwd)

if [[ ! -f Makefile || ! -d src/physics ]]; then
  echo "error: run this from the repository root" >&2
  exit 1
fi
if [[ ! -d "$RES" ]]; then
  echo "error: results directory '$RES' not found (set RESULTS_DIR)" >&2
  exit 1
fi

echo "==> staging into $OUT/  (results from $RES)"
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT"

# --- code, gates, drivers -------------------------------------------------
cp -r src apps test "$OUT/"
find "$OUT/src" "$OUT/apps" "$OUT/test" \( -name '*.bak' -o -name '*~' -o -name '* copy*' \) -delete
cp Makefile setup_env.src "$OUT/"

# --- scripts (post-processing + job submission) ---------------------------
mkdir -p "$OUT/scripts"
cp scripts/make_figures.py scripts/make_showcase_figs.py scripts/beji_compare.py \
   scripts/make_tsunami_geometry.py scripts/make_tsunami_field.py \
   scripts/collect_sweep.py scripts/collect_ncu.py \
   scripts/bench.sbatch scripts/submit_bench.sh \
   scripts/shoal_focus.sbatch scripts/shoal_sweep.sbatch scripts/tsunami.sbatch \
   scripts/profile_nsys.sbatch scripts/profile_ncu.sbatch \
   scripts/make_release.sh \
   "$OUT/scripts/"

# --- repository metadata ----------------------------------------------------
# docs/ holds the authors' own guides and working documents (in Spanish) and is
# deliberately not shipped; the README and the comments in every source file
# document the code.
cp release/README.md release/MANIFEST.md release/LICENSE release/CITATION.cff "$OUT/"
cp release/gitignore "$OUT/.gitignore"

# --- digitized laboratory data --------------------------------------------
mkdir -p "$OUT/data"
cp "$RES/beji_lab_harmonics.csv" "$OUT/data/" 2>/dev/null || true
if [[ -d "$RES/beji_extracted_results" ]]; then
  cp -r "$RES/beji_extracted_results" "$OUT/data/beji_extracted"
fi

# --- reference results (no VTK, no raw profiler binaries) ------------------
mkdir -p "$OUT/results"
find "$RES" -maxdepth 1 -type f ! -name '*.vtk' ! -name '*.nsys-rep' \
     ! -name '*.sqlite' ! -name '*.ncu-rep' ! -name '*.bak' \
     -exec cp {} "$OUT/results/" \;

# --- provenance -----------------------------------------------------------
# Recorded only where the toolchain is present (MN5); elsewhere the toolchain
# of the reported runs is the one documented in MANIFEST.md.
if command -v nvcc >/dev/null 2>&1; then
  {
    echo "# Build environment recorded at packaging time"
    echo "host:  $(hostname)"
    echo "date:  $(date -u +%FT%TZ)"
    echo "nvcc:  $(nvcc --version 2>/dev/null | tail -1 || echo 'not available')"
    echo "mpi:   $(mpirun --version 2>&1 | head -1 || echo 'not available')"
    echo "gcc:   $(gcc --version 2>/dev/null | head -1 || echo 'not available')"
    echo "python:$(python3 --version 2>&1 || echo ' not available')"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
      | head -1 | sed 's/^/gpu:   /' || true
  } > "$OUT/BUILD_ENVIRONMENT.txt"
fi

# --- archive --------------------------------------------------------------
echo "==> creating $OUT.tar.gz"
tar czf "$OUT.tar.gz" "$OUT"

echo
echo "package:   $ROOT/$OUT"
echo "archive:   $ROOT/$OUT.tar.gz"
echo "size:      $(du -sh "$OUT" | cut -f1) unpacked, $(du -h "$OUT.tar.gz" | cut -f1) compressed"
echo "files:     $(find "$OUT" -type f | wc -l)"
