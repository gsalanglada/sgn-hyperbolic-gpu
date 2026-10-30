# Beji/Battjes - extracted reference package

Contents:

- data/beji_lab_harmonics.csv
  Approximate harmonic amplitudes for condition A, columns:
  x_m, A1_cm, A2_cm, A3_cm, mean_eta_cm, source.

- data/beji_lab_conditionA_timeseries_digitized.csv
  Approximate digitized free-surface elevation time series from Dingemans (1994), Fig. 3 right panels,
  resampled to a common two-period time grid. Units: time_s and eta in cm.

- data/beji_lab_conditionA_timeseries_digitized_long.csv
  Same digitized signal in long/tidy format at the native extraction resolution.

- figures/*.png
  Cropped figures useful for qualitative comparison and setup verification.

- scripts/digitize_conditionA_from_fig3.py
  Reproducible extraction script used to generate the CSVs.

Important caveat:
The PDFs do not contain numeric raw laboratory files. The CSV data here are approximate values obtained by digitizing the plotted time-window in Dingemans (1994), Fig. 3, condition A. They are useful for overlays and sanity checks, but should not be treated as original laboratory data.
