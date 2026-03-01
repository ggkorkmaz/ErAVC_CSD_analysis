# ErAVC CSD Analysis

This repository contains a Python script used for crystal size distribution (CSD) analysis and residence time estimation in the ErAVC magma recharge study.

## File

- **ErAVC_CSD_analysis.py**  
  Performs CSD computation, linear and piecewise regression, AIC comparison, bootstrap breakpoint confidence intervals, residence time calculation, and figure generation.

## Requirements

- numpy
- pandas
- matplotlib
- scipy

## Data

Input files must contain a column named `Feret` (µm).
