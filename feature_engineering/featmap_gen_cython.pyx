# featmap_gen_cython.pyx
import numpy as np
import pandas as pd
cimport numpy as cnp  # Correct way to import NumPy for Cython optimization

def featmap_gen_cython(df):
    cdef int n = len(df)
    cdef int i, j
    cdef float temp_time, temp_amt
    # Using NumPy arrays for efficient calculations
    cdef cnp.ndarray[cnp.float32_t, ndim=1] time_array = df['Time'].values.astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] amount_array = df['Amount'].values.astype(np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] time_span = np.array([2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000], dtype=np.float32)
    cdef list time_name = [str(int(i)) for i in time_span]

    results = []

    for i in range(n):
        temp_time = time_array[i]
        temp_amt = amount_array[i]
        temp_results = {}

        for j, length in enumerate(time_span):
            tname = time_name[j]
            lowbound = time_array >= (temp_time - length)
            upbound = time_array <= temp_time
            indices = np.where(lowbound & upbound)[0]  # Using NumPy for where function
            correct_data = df.iloc[indices]

            # Compute features
            temp_results[f'trans_at_avg_{tname}'] = correct_data['Amount'].mean() if not correct_data.empty else 0
            temp_results[f'trans_at_totl_{tname}'] = correct_data['Amount'].sum() if not correct_data.empty else 0
            temp_results[f'trans_at_std_{tname}'] = correct_data['Amount'].std() if not correct_data.empty else 0
            temp_results[f'trans_at_bias_{tname}'] = temp_amt - correct_data['Amount'].mean() if not correct_data.empty else 0
            temp_results[f'trans_at_num_{tname}'] = len(correct_data)
            temp_results[f'trans_target_num_{tname}'] = len(correct_data['Target'].unique()) if 'Target' in correct_data.columns else 0
            temp_results[f'trans_location_num_{tname}'] = len(correct_data['Location'].unique()) if 'Location' in correct_data.columns else 0
            temp_results[f'trans_type_num_{tname}'] = len(correct_data['Type'].unique()) if 'Type' in correct_data.columns else 0

        results.append(temp_results)

    return pd.DataFrame(results)