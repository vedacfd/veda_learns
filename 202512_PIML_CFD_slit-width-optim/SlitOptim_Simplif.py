#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import glob
from scipy.optimize import minimize, least_squares

# Engineering Limitations (manufacturability, precision related)
MIN_WIDTH = 0.0  #mm, minimum slit width allowable
MAX_WIDTH = 0.0  #mm, maximum slit width allowable
STEP_SIZE = 0.0  #mm, for eg: 5mm increments search
SLIT_ORDER = [f'L{i}' for i in range(10)] + [f'R{i}' for i in range(1, 10)]

# Target Definitions
LOW_FLOW_SLITS = ['L0', 'L9', 'R1', 'R9']
LOW_FLOW_TARGET = 0.0 #ideal flow percentage target
HIGH_FLOW_TARGET = 0.0 #ideal flow percentage target

# Load data
def load_data():
    files = sorted(glob.glob('case*.csv'))
    if not files:
        print("No case files found. Check directory.")
        return None

    print(f"Loading {len(files)} case files.")
    all_cases = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns] # Clean headers

            # Extract data
            case_map = {}
            for _, row in df.iterrows():
                # Clean slit name and value
                s_name = row['Slit'].strip()
                w_val = float(row['Slit Width'])
                f_str = str(row['Flowrate Percent']).replace('%', '') # my original csv had percentage mark following the value
                f_val = float(f_str)
                case_map[s_name] = (w_val, f_val)

            widths = []
            flows = []
            valid = True
            for s in SLIT_ORDER:
                if s in case_map:
                    widths.append(case_map[s][0])
                    flows.append(case_map[s][1])
                else:
                    print(f"Warning: Slit {s} missing in {f}")
                    valid = False

            if valid:
                all_cases.append({
                    'widths': np.array(widths),
                    'flows': np.array(flows)
                })
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return all_cases

# Physics model
def predict_flow(widths, C, alpha):
    # Model: Flow ~ C * Width^alpha
    # Normalized so that sum(Flow) = 100

    # Protect against 0 width errors
    w_safe = np.maximum(widths, 1e-9)
    conductance = C * (w_safe ** alpha)
    total_conductance = np.sum(conductance)
    return 100.0 * conductance / total_conductance

# Calibration (Learn alpha and C)
def train_model(cases): # Goal is to find 1 value of alpha and 19 values of C (C0 fixed to 1) that minimizes error across all cases

    def loss_func(params):
        alpha = params[0]
        C = np.concatenate(([1.0], params[1:]))

        residuals = []
        for c in cases:
            pred = predict_flow(c['widths'], C, alpha)
            residuals.extend(pred - c['flows'])
        return np.array(residuals)

    # Initial guess
    x0 = np.ones(19) 
    # Bounds: alpha [0.1, 4.0], C [0, inf]
    lb = [0.1] + [0.0]*18
    ub = [4.0] + [np.inf]*18

    print("Calibrating physics model on historical data")
    res = least_squares(loss_func, x0, bounds=(lb, ub), method='trf')

    best_alpha = res.x[0]
    best_C = np.concatenate(([1.0], res.x[1:]))

    # Calculate fit quality (RMSD of the model vs history)
    mse = np.mean(res.fun**2)
    rmsd = np.sqrt(mse)
    print(f"Model calibrated. Historical Fit RMSD: {rmsd:.4f}")
    print(f"Learned Flow Exponent (alpha): {best_alpha:.4f}")

    return best_alpha, best_C

# Optimization (Minimizing RMSD)
def calculate_rmsd(pred, target):
    return np.sqrt(np.mean((pred - target)**2))

def optimize_widths(target, alpha, C): #continuous optimization

    def objective(w): # Minimize RMSD
        pred = predict_flow(w, C, alpha) 
        return calculate_rmsd(pred, target)

    bounds = [(MIN_WIDTH, MAX_WIDTH) for _ in range(19)]
    w0 = np.full(19, 40.0) # Start flat

    res = minimize(objective, w0, bounds=bounds, method='L-BFGS-B')
    w_cont = res.x

    #Round to step size depending on precision input; discrete optimization
    w_disc = STEP_SIZE * np.round(w_cont / STEP_SIZE)
    w_disc = np.clip(w_disc, MIN_WIDTH, MAX_WIDTH)

    # Because rounding the continuous best doesn't always equal the discrete best (due to the normalization constraint), i.e., changing one slit affects all others
    print("Improving solution based on set precision limit")

    current_w = w_disc.copy()
    current_rmsd = calculate_rmsd(predict_flow(current_w, C, alpha), target)

    improved = True
    while improved:
        improved = False
        best_local_w = current_w.copy()
        best_local_rmsd = current_rmsd

        # Try moving every slit ±step_size
        for i in range(19):
            original_val = current_w[i]

            # Try options: current, + step_size, - step_size
            options = []
            if original_val + STEP_SIZE <= MAX_WIDTH:
                options.append(original_val + STEP_SIZE)
            if original_val - STEP_SIZE >= MIN_WIDTH:
                options.append(original_val - STEP_SIZE)

            for opt_val in options:
                test_w = current_w.copy()
                test_w[i] = opt_val

                test_pred = predict_flow(test_w, C, alpha)
                test_rmsd = calculate_rmsd(test_pred, target)

                if test_rmsd < best_local_rmsd - 1e-6: # Small epsilon
                    best_local_rmsd = test_rmsd
                    best_local_w = test_w.copy()
                    improved = True

        if improved:
            current_w = best_local_w
            current_rmsd = best_local_rmsd

    return current_w

# MAIN
if __name__ == "__main__":
    cases = load_data()

    if cases:
        # 1. Train
        alpha, C = train_model(cases)

        # 2. Setup Target
        target = np.zeros(19)
        for i, s in enumerate(SLIT_ORDER):
            if s in LOW_FLOW_SLITS:
                target[i] = LOW_FLOW_TARGET
            else:
                target[i] = HIGH_FLOW_TARGET

        target = 100.0 * target / np.sum(target)

        # Optimize
        best_widths = optimize_widths(target, alpha, C)

        # Result
        pred_flow = predict_flow(best_widths, C, alpha)

        # Calculate RMSD Breakdown
        diffs = pred_flow - target

        # Indices
        low_idxs = [i for i, s in enumerate(SLIT_ORDER) if s in LOW_FLOW_SLITS]
        high_idxs = [i for i, s in enumerate(SLIT_ORDER) if s not in LOW_FLOW_SLITS]

        rmsd_total = np.sqrt(np.mean(diffs**2))
        rmsd_low = np.sqrt(np.mean(diffs[low_idxs]**2))
        rmsd_high = np.sqrt(np.mean(diffs[high_idxs]**2))

        df_out = pd.DataFrame({
            'Slit': SLIT_ORDER,
            'Width (mm)': best_widths.astype(int),
            'Predicted %': np.round(pred_flow, 2),
            'Target %': np.round(target, 2),
            'Deviation': np.round(diffs, 2)
        })

        print("\n")
        print("Optimized Result")
        print("\n")
        print(df_out.to_string(index=False))
        print("\n")
        print(f"Global RMSD:      {rmsd_total:.4f}")
        print(f"Low Flow RMSD:    {rmsd_low:.4f} (Slits: {', '.join(LOW_FLOW_SLITS)})")
        print(f"High Flow RMSD:   {rmsd_high:.4f} (Remaining 15 slits)")
        print("\n")

        # Save
        next_id = len(cases) + 1
        fname = f"case{next_id}.csv"

        # Save in original CSV format
        save_df = df_out[['Slit', 'Width (mm)']].copy()
        save_df.columns = ['Slit', 'Slit Width']
        save_df['Flowrate Percent'] = 0.0

        save_df.to_csv(fname, index=False)
        print(f"Saved to {fname}. Run simulation, update Flowrate, and re-run model until best-case obtained.")


# In[ ]:




