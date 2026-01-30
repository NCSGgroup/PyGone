# PyGone: Python codes for Gravity satellite level-one data processing
A Python toolbox exclusively for GRACE(-FO) Level-1A data processing, which currently supports (1) AOD1B modeling and (2) attitude determination. Be aware that this toolbox is still being developed, since
more functionalities are planned to address the processing of a complete set of level-1a instrument: KBR, LRI, ACC, and GNSS on board GRACE(-FO).

by Fan Yang (fany@plan.aau.dk), Xudong Pan (), Weihang Zhang (zwh_cge@hust.edu.cn)
***

### Module-1: AOD1B modelling and atmospheric tide modelling

It can be used for (1) establishing AOD1B product, (2) tide harmonic analysis, (3) vertical water vapor computation.

Related publications:
1. Yang, F., Bai, J., Liu, H., Zhang, W., Wu, Y., Liu, S., Shi, C., Zhang, T., Zhong, M., Zhu, Z., Wang, C., Forootan, E., Yu, J., Yu, Z., and Xiao, Y.: CRA-LICOM: a global high-frequency atmospheric and oceanic temporal gravity field product (2002–2024), Earth Syst. Sci. Data, 17, 4691–4714, https://doi.org/10.5194/essd-17-4691-2025, 2025.
2. Zhang, W., Yang, F., Wu, Y. et al. HUST-CRA: A New Atmospheric De-Aliasing Model for Satellite Gravimetry. Adv. Atmos. Sci. 42, 382–396 (2025). https://doi.org/10.1007/s00376-024-4045-6
3. Yang, F., Forootan, E., Wang, C., Kusche, J., & Luo, Z. (2021). A new 1-hourly ERA5-based atmosphere de-aliasing product for GRACE, GRACE-FO, and future gravity missions. Journal of Geophysical Research: Solid Earth, 126, e2021JB021926. https://doi.org/10.1029/2021JB021926
***

### Module-2: Precise attitude determination from onboard star camera and IMU
This module supports: (1) raw data processing of IMU, SCA and other relevant Level-1A instrument data; 
(2) implementing the statistically optimized fusion of the multiple SCAs on board.
(3) enabling the Kalman filter for the fusion of SCA and IMU towards a more precise attitude determination.

Related publications:
1. X. Pan, F. Yang and Y. Wu, "A Statistically Optimized Star Camera Fusion Approach for the Attitude Determination of Low-Low Satellite-to-Satellite Tracking Mission," in IEEE Sensors Journal, vol. 25, no. 15, pp. 28800-28814, 1 Aug.1, 2025, doi: 10.1109/JSEN.2025.3578622.
2. Yang, F.; Liang, L.; Wang, C.; Luo, Z. Attitude Determination for GRACE-FO: Reprocessing the Level-1A SC and IMU Data. Remote Sens. 2022, 14, 126. https://doi.org/10.3390/rs14010126
