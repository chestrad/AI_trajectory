#Code was generated using ChatGPT (GPT-4; OpenAI).
#spline interpolation for missing inputs

import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np

# Load your data
data = pd.read_excel('for interpolation.xlsx')

# Function for spline interpolation
def spline_interpolate(row):
    time_points = [0, 1, 2, 3]  # baseline, T1, T2, T3
    valid_times = [time_points[i] for i in range(len(row)) if not np.isnan(row[i])]
    valid_values = [row[i] for i in range(len(row)) if not np.isnan(row[i])]

    if len(valid_times) > 2:
        cs = CubicSpline(valid_times, valid_values)
        return [cs(t) for t in time_points]
    else:
        return np.interp(time_points, valid_times, valid_values)

# Apply spline interpolation
spline_interpolated_data = data.set_index('ID').apply(spline_interpolate, axis=1, result_type='expand')
spline_interpolated_data.columns = ['baseline', 'T1', 'T2', 'T3']
