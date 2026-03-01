# ErAVC CSD Analysis

This repository contains Python scripts used for crystal size distribution (CSD) analysis and residence time estimation in the ErAVC magma recharge study.

## Files

- **csd_analysis.py**  
  Performs CSD computation, linear and piecewise regression, AIC comparison, bootstrap breakpoint confidence intervals, and residence time calculation.

- **csd_plotting.py**  
  Reproduces publication-quality CSD figures using the computed results.

## Requirements

- numpy
- pandas
- matplotlib
- scipy

## Data

Input files must contain a column named `Feret` (µm).
