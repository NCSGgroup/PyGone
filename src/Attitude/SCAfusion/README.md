# GRACE-FO Star Camera and IMU Attitude Determination

This repository is designed for processing GRACE/GRACE-FO star camera (SCA), IMU, and other Level-1A/1B data. It includes multi-star-camera attitude fusion, attitude geometry, spectral and pointing analysis, and visualization tools.

## Code Organization

```
pysrc/
  core/         Task enums, coordinate systems, and numerical utilities
  data/         Level-1A/1B instrument data readers and data collection
  imu/          IMU data preprocessing
  attitude/     LOS, pointing, and attitude geometry
  fusion/       Star camera fusion, model variants, and filter extension points
    models/     Star camera fusion model variants
  analysis/     HFN, correlation, spectral, solar angle, and camera analysis
    plots/      Result plots and publication figures
  experiments/  Standalone experiments and demonstration scripts
  tools/        Small maintenance utilities
  getSunPos/    Solar position and reference-frame subsystem
                (preserves its original internal structure)
```

## Algorithmic Workflow Boundaries

Sensor data are independently processed through Level-1A decoding, time correction, outlier and sign handling, interpolation, and coordinate-frame transformations before entering two independently usable fusion paths:

```
Raw L1A/L1B Data
       |
       v
Data Reading and Time/Coordinate Preprocessing
       |
       +------------------------------+
       |                              |
       v                              v
Star Camera Fusion Path         Multi-Sensor Fusion Path
       |                              |
Multi-SCA covariance            Individual SCA observations
modeling/optimization           + Preprocessed IMU angular rates
       |                              |
Quaternion GLS Fusion           Indirect Kalman Filtering
       |                              |
       v                              v
Star Camera Attitude Product    Attitude Product
```

These two paths should not be forced into a serial architecture.

The GRACE-FO method described in the second paper incorporates individual star camera observations into the filter and dynamically adjusts their weights. Therefore, it should not be assumed that every Kalman workflow must first generate a fixed-weight combined star camera observation.

The 2025 star camera fusion study emphasizes dynamic, non-uniform noise covariance and quaternion Generalized Least Squares (GLS), using GNSS orbit information as an auxiliary prior.

The specific processing path should be determined by the algorithm configuration rather than inferred solely from a module name such as "multi-star-camera fusion."

Organizing files by responsibility does not imply that every processing stage has already been implemented as a fully executable end-to-end pipeline. Input configuration, product output paths, and existing research scripts still partly depend on the working directory or historical directory structures.

## Current Implementation Status

- `pysrc.data.instruments`: Existing L1A/L1B instrument data readers.
- `pysrc.imu.preprocessing`: The existing IMU preprocessing class is retained. Its code is currently a skeleton and should not be considered a complete IMU processing pipeline.
- `pysrc.fusion.multi_sca` and `pysrc.fusion.models`: Existing multi-SCA fusion implementations and model variants. This reorganization only relocates the modules; it does not verify whether they fully reproduce the GNSS-assisted GLS method described in the 2025 paper.
- `pysrc.fusion.kalman`: Explicitly marked as `not_implemented`. The legacy `kalman.py` contains comments only. This reorganization establishes its module location without inventing a Kalman algorithm.
- `pysrc.analysis`: Existing analysis, diagnostic, and visualization code. Some legacy scripts still use historical data paths and interfaces.

Before implementing Kalman fusion, the following aspects must be clearly defined:

1. The definitions of IMU angular increments and angular rates.
2. Reference-frame and time-synchronization conventions between SCA and IMU.
3. The error-state formulation.
4. Initial conditions.
5. Gyroscope bias and noise covariance.
6. Whether the Kalman path receives individual SCA observations or uses pre-fused observations selected through configuration.

The filter conventions and noise parameters directly determine the results and cannot be inferred from directory descriptions or paper titles.

## Import Conventions

New code should use the layered import paths, for example:

```
from pysrc.data.instruments import GetInstrument_L1A, GetInstrument_L1B
from pysrc.fusion.multi_sca import SCAweighting
from pysrc.core.settings import Mission, SatID
```

Legacy import paths are retained as compatibility shims, such as:

- `pysrc.GetInstrument`
- `pysrc.SCAweighting`
- `pysrc.Setting`

New code should not continue to depend on these top-level compatibility modules. Existing consumers can migrate gradually to the new layered import structure.

## Data and Validation Scope

The following directories in the repository root contain data, products, or results and were not moved or modified during this reorganization:

- `data/`
- `CA/`
- `FusionProduct/`
- `HFN_Euler/`
- `KBR_result/`
- `LOS/`
- `Produce/`
- `SunAngle/`
- `paper_figure/`

The repository currently does not include:

- A unified project dependency manifest.
- A configuration-driven main processing pipeline.
- An existing ready-to-use test suite.

Post-reorganization checks mainly focus on source-file syntax parsing, package and legacy entry-point mapping, and static import-path checks.

Scientific results using real instrument data have not been recomputed as part of this reorganization.

## References

1. X. Pan, F. Yang, and Y. Wu, "A Statistically Optimized Star Camera Fusion Approach for the Attitude Determination of Low-Low Satellite-to-Satellite Tracking Mission," *IEEE Sensors Journal*, 25(15), 28800–28814, 2025. DOI: [10.1109/JSEN.2025.3578622](https://doi.org/10.1109/JSEN.2025.3578622).
2. F. Yang, L. Liang, C. Wang, and Z. Luo, "Attitude Determination for GRACE-FO: Reprocessing the Level-1A SC and IMU Data," *Remote Sensing*, 14(1), 126, 2022. DOI: [10.3390/rs14010126](https://doi.org/10.3390/rs14010126).
