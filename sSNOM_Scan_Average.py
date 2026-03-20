import numpy as np
import pandas as pd
import h5py
import cv2
from scipy.optimize import leastsq, curve_fit
from scipy.ndimage import median_filter
import glob
import re
import matplotlib.pyplot as plt
import os, copy

# ==========================================
# AFM Topography Data Loading & Formatting
# ==========================================

def load_scan_files(folder_path):
    """
    Loads AFM topography CSV files, sorts them by scan number, and extracts metadata.
    """
    # Find all scan CSVs and sort them numerically based on the integer in the filename
    scan_files = sorted(glob.glob(os.path.join(folder_path, "scan_*.csv")), key=lambda x: int(re.search(r'scan_(\d+)_', x).group(1)))
    
    # Read the first row of the first file to get scan constants/metadata
    df_consts = pd.read_csv(scan_files[0], nrows=1, header=None, names=["image_width", "image_height", "time_line", "points_line", "lines", "rotation"])
    
    scan_data = []
    for file in scan_files:
        # Skip the header row and load the actual scan data points
        df = pd.read_csv(file, skiprows=1, header=None, names=["x", "y", "topo_f", "topo_b", "amp_f", "amp_b"])
        scan_data.append(df)
    return scan_data, df_consts

def get_scan_shape(scan_df):
    """Calculates the 2D grid dimensions (rows, cols) from the unique 1D X and Y coordinates."""
    unique_x = np.unique(scan_df['x']).size
    unique_y = np.unique(scan_df['y']).size
    return unique_y, unique_x

def get_aspect(data):
    """Calculates the aspect ratio (width/height) of a 2D array for plotting."""
    r,c = np.shape(data)
    return c/r  

# ==========================================
# Image Processing & Artifact Correction
# ==========================================

def subtract_background_plane(image):
    """
    Removes a first-order global tilt (background plane) from an image using least squares.
    This is standard practice to flatten AFM topography data.
    """
    y, x = np.indices(image.shape)
    # Create design matrix for a plane: z = ax + by + c
    A = np.c_[x.ravel(), y.ravel(), np.ones_like(x).ravel()]
    B = image.ravel()
    
    # Solve for plane coefficients
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    plane = (coeffs[0] * x + coeffs[1] * y + coeffs[2]).astype(image.dtype)
    return image - plane

def align_images(reference, target):
    """
    Aligns a target image to a reference image using Enhanced Correlation Coefficient (ECC) maximization.
    This corrects for sample drift between consecutive AFM scans.
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32) # Initial guess: identity matrix (no shift)
    # Define termination criteria: 5000 iterations or epsilon of 1e-10
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5000, 1e-10)
    
    # Find the Euclidean transform (translation + rotation) that aligns the images
    warp_matrix = cv2.findTransformECC(reference.astype(np.float32), target.astype(np.float32), warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)[1]
    
    height, width = reference.shape
    # Apply the calculated warp matrix to the target image
    aligned_target = cv2.warpAffine(target, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    
    # Create a mask to track which pixels are valid and which are padded artifacts from the shift
    mask = cv2.warpAffine(np.ones_like(target, dtype=np.uint8), warp_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP)
    
    # Scale the translations in the warp matrix to relative coordinates (0 to 1) 
    # so they can be applied to arrays of different resolutions (e.g., the optical data)
    warp_matrix_scaled = warp_matrix.copy()
    warp_matrix_scaled[0, 2] /= width
    warp_matrix_scaled[1, 2] /= height
    
    return aligned_target, warp_matrix_scaled, mask

def fit_plane_and_shift(data):
    """
    Alternative plane fitting function using scipy.leastsq.
    Flattens the image and shifts the entire dataset so the minimum Z value is 0.
    """
    rows, cols = data.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    X, Y, Z = X.ravel(), Y.ravel(), data.ravel()
    
    def plane(params, x, y):
        a, b, c = params
        return a * x + b * y + c
    
    def error(params, x, y, z):
        return plane(params, x, y) - z
    
    initial_guess = [0, 0, np.mean(Z)]  
    best_fit, _ = leastsq(error, initial_guess, args=(X, Y, Z))  
    
    plane_values = plane(best_fit, X, Y).reshape(rows, cols)  
    adjusted_data = data - plane_values  
    adjusted_data -= np.min(adjusted_data)  # Shift minimum to 0
    return adjusted_data

def fit_poly_col_and_shift(data, fit_flag=None, *args):
    """
    Fits and removes line-by-line artifacts (often called 'scanner bow') along columns.
    """
    rows, cols = data.shape
    X = np.arange(rows)  
    
    # Define quintic function (5th order polynomial)
    def quintic(x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

    # Determine which part of the data to use as the reference profile for the fit
    if fit_flag in ["avg", "centre", "edges", "left", "right", "known"]:
        if fit_flag == "avg": Y = np.mean(data, axis=1)
        elif fit_flag == "centre": Y = data[:, cols//2]
        elif fit_flag == "edges": Y = np.mean(data[:,[3,-4]], axis=1) # Average of specific left/right edge columns
        elif fit_flag == "left": Y = data[:,3]
        elif fit_flag == "right": Y = data[:,-4]

        # Fit the profile
        initial_guess = [0, 0, 0, 0, 0, np.mean(Y)]  
        popt, _ = curve_fit(quintic, X, Y, p0=initial_guess)  

        # If we explicitly pass in known parameters, override the fit
        if fit_flag == "known": popt = args[0]

        # Subtract the fitted 1D profile from every single column in the 2D array
        data = np.subtract(data.T, quintic(X, *popt)).T  

    data -= np.min(data)  # Shift minimum to 0
    return data, popt

def avg_w_mask(data_list, mask_list, med_filt=1):
    """
    Averages a list of aligned arrays, ignoring invalid pixels (zeros in the mask) 
    created during the image shifting process.
    """
    data = np.array(data_list)
    mask = np.array(mask_list)
    
    # Create a masked array where mask==False is ignored in calculations
    mdata = np.ma.masked_array(data, mask=~mask.astype(bool))
    avg_data = np.ma.average(mdata, axis=0).filled(0)  
    std_data = np.ma.std(mdata, axis=0).filled(0)  
    
    if med_filt: avg_data = median_filter(avg_data, size=(3,3)) # Optional 3x3 median filter for noise reduction
    return avg_data, std_data

# ==========================================
# Zurich Lock-In (HDF5) Data Processing
# ==========================================

def load_h5_file(h5_filename):
    """Basic HDF5 loader for Zurich instrument demodulator data (X and Y components)."""
    with h5py.File(h5_filename, 'r') as h5f:
        j_values = np.array(sorted(map(int, h5f.keys())))
        channels = list(h5f[f'{j_values[0]:03}/dev1170/demods'].keys())

        data_shape = np.shape(np.array(h5f[f'{j_values[0]:03}/dev1170/demods/{channels[0]}/sample.x.avg/value']))
        data = np.zeros([len(j_values), len(channels), 2, *data_shape])

        for jdx, j in enumerate(j_values):
            for idx, channel in enumerate(channels):
                data[jdx, idx, 0] = np.array(h5f[f'{j:03}/dev1170/demods/{channel}/sample.x.avg/value'])
                data[jdx, idx, 1] = np.array(h5f[f'{j:03}/dev1170/demods/{channel}/sample.y.avg/value'])

        return data, channels

def process_h5_file(data, channels, scan_shape, shifts, img_include, align):
    """
    Converts lock-in X/Y signals to Amplitude (R) and Phase (Theta), de-interlaces 
    forward/backward scan lines, and applies the drift-correction shifts calculated from Topography.
    """
    imgs = {channel: {'R_f': [], 'R_b': [], 'theta_f': [], 'theta_b': []} for channel in channels}
    masks = {channel: {'R_f': [], 'R_b': [], 'theta_f': [], 'theta_b': []} for channel in channels}
    
    height_topo, width_topo = scan_shape

    # Pad inclusion boolean array to match data dimensions
    img_include = np.pad(img_include, (0, data.shape[0]-len(img_include)))
    j_values = np.array(list(range(data.shape[0])))
    j_values = j_values[np.array(img_include, dtype=np.bool)] 

    for jdx, j in enumerate(j_values):
        for idx, channel in enumerate(channels):
            x_data = data[jdx, idx, 0]
            y_data = data[jdx, idx, 1]
            
            # 1. De-interlace: AFM records forward (left-to-right) and backward (right-to-left) lines alternately
            forward_x, backward_x = x_data[::2], x_data[1::2] 
            forward_y, backward_y = y_data[::2], y_data[1::2]
            
            # 2. Trim: Drop initial settling lines (repeats at the top of the scan area)
            forward_x, backward_x = forward_x[-height_topo:], backward_x[-height_topo:] 
            forward_y, backward_y = forward_y[-height_topo:], backward_y[-height_topo:]
            
            height_sig, width_sig = forward_x.shape

            # 3. Convert Cartesian lock-in data (X,Y) to Polar (Amplitude/R, Phase/Theta)
            R_f, R_b = np.hypot(forward_x, forward_y), np.hypot(backward_x, backward_y)
            theta_f, theta_b = np.arctan2(forward_y, forward_x)*180/np.pi, np.arctan2(backward_y, backward_x)*180/np.pi

            R_f_mask, theta_f_mask, R_b_mask, theta_b_mask = [np.ones_like(R_f, dtype=np.uint8) for _ in range(4)]
            
            # 4. Apply Spatial Drift Alignment
            if jdx > 0 and align:
                # Retrieve the scaled warp matrix from Topography, scale it to the optical image resolution
                shift = copy.deepcopy(shifts[jdx - 1])
                shift[0,2] *= width_sig
                shift[1,2] *= height_sig

                # Apply warp to Amplitude and Phase arrays, and calculate masks for padded edges
                R_f = cv2.warpAffine(R_f, shift, (width_sig, height_sig), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                R_f_mask = cv2.warpAffine(R_f_mask, shift, (width_sig, height_sig), flags=cv2.WARP_INVERSE_MAP)
                # ... (repeat for theta_f, R_b, theta_b) ...
                theta_f = cv2.warpAffine(theta_f, shift, (width_sig, height_sig), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                theta_f_mask = cv2.warpAffine(theta_f_mask, shift, (width_sig, height_sig), flags=cv2.WARP_INVERSE_MAP)
                R_b = cv2.warpAffine(R_b, shift, (width_sig, height_sig), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                R_b_mask = cv2.warpAffine(R_b_mask, shift, (width_sig, height_sig), flags=cv2.WARP_INVERSE_MAP)
                theta_b = cv2.warpAffine(theta_b, shift, (width_sig, height_sig), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
                theta_b_mask = cv2.warpAffine(theta_b_mask, shift, (width_sig, height_sig), flags=cv2.WARP_INVERSE_MAP)               
                
            imgs[channel]['R_f'].append(R_f)
            masks[channel]['R_f'].append(R_f_mask)
            # ... appending other arrays to dictionaries ...
            imgs[channel]['R_b'].append(R_b)
            masks[channel]['R_b'].append(R_b_mask)
            imgs[channel]['theta_f'].append(theta_f)
            masks[channel]['theta_f'].append(theta_f_mask)
            imgs[channel]['theta_b'].append(theta_b)
            masks[channel]['theta_b'].append(theta_b_mask)

    # 5. Calculate Final Averages and Standard Deviations across all scans
    results = {channel: {'R_f': np.array(0), 'R_b': np.array(0), 'theta_f': np.array(0), 'theta_f_std': np.array(0), 'theta_b': np.array(0), 'theta_b_std': np.array(0)} for channel in channels}
    
    for idx, channel in enumerate(channels):
        results[channel]['R_f'], _ = avg_w_mask(imgs[channel]['R_f'], masks[channel]['R_f'])
        results[channel]['R_b'], _ = avg_w_mask(imgs[channel]['R_b'], masks[channel]['R_b'])
        results[channel]['theta_f'], results[channel]['theta_f_std'] = avg_w_mask(imgs[channel]['theta_f'], masks[channel]['theta_f'], med_filt=0) # No median filter on phase to preserve sharp boundaries
        results[channel]['theta_b'], results[channel]['theta_b_std'] = avg_w_mask(imgs[channel]['theta_b'], masks[channel]['theta_b'], med_filt=0)
        
    return results

def get_sorted_files_from_numbered_folders(base_dir, folder_prefix="zurich_"):
    """Finds folders (e.g., 'zurich_1') and extracts their inner file paths in sorted numerical order."""
    found_files = []
    folder_pattern = re.compile(fr'^{folder_prefix}(\d+)$')

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            match = folder_pattern.match(entry)
            if match:
                folder_number = int(match.group(1))
                files_in_folder = os.listdir(full_path)
                if len(files_in_folder) == 1:
                    file_name = files_in_folder[0]
                    file_path = os.path.join(full_path, file_name)
                    found_files.append((folder_number, file_path))
                
    found_files.sort(key=lambda x: x[0])
    sorted_paths = [path for number, path in found_files]
    return sorted_paths

def load_h5_autosave_avg(folder_path, params):
    """ 
    Advanced HDF5 Loader for Zurich 'Autosave' files. 
    Crucial for s-SNOM: Maps high-frequency lock-in amplifier samples down to specific physical pixels.
    Params:
    - px: desired x pixels
    - scan_t_N: Nanite AFM scan line time (s)
    - scan_t_Z: Zurich scan line time (s)
    - tc: lock-in time constant (s)
    - settling: settling multiplier (usually 4-5x tc to reach 99% of signal)
    """
    h5_files = sorted(glob.glob(os.path.join(folder_path, "*_autosave_*.h5")), key=lambda x: int(re.search(r'.*_autosave_(\d+)', x).group(1)))
    num_h5_scans = 0
    num_h5_channels = 0
    channels = []
    data_shape = np.array(0)

    # First pass: determine dataset shape and total number of scans/channels
    for idx, file in enumerate(h5_files):
        with h5py.File(file, 'r') as h5f:
            h5_scan_keys = list(h5f.keys())
            num_h5_scans += len(h5_scan_keys)
            if not idx:
                h5_channel_keys = list(h5f[f'{h5_scan_keys[0]:03}/dev1170/demods'].keys())
                num_h5_channels = len(h5_channel_keys)
                for jdx in h5_channel_keys:
                    try:
                        data_shape = np.shape(np.array(h5f[f"{h5_scan_keys[0]:03}/dev1170/demods/{jdx}/sample.x.avg/value"]))
                        channels.append(jdx)
                    except KeyError:
                        continue
                        
    px, scan_t_N, scan_t_Z, tc, settling = params[:5]
    data = np.zeros([num_h5_scans, num_h5_channels, 2, data_shape[0], px])
    
    # Calculate alignment parameters
    sample_rate = data_shape[1]/scan_t_Z
    desired_len = int(scan_t_N*sample_rate/px)*px # Total valid samples (divisible by pixel count)
    settling_px = int(tc*settling*sample_rate) # Number of samples to discard due to settling

    scan_idx = 0
    # Second pass: Extract, reshape, trim settling time, and average to desired pixels
    for fdx, file in enumerate(h5_files):
        with h5py.File(file, 'r') as h5f:
            for sdx, scan_key in enumerate(h5f.keys()):
                for cdx, channel in enumerate(channels):
                    data_x_raw = np.array(h5f[f"{scan_key}/dev1170/demods/{channel}/sample.x.avg/value"])   
                    data_y_raw = np.array(h5f[f"{scan_key}/dev1170/demods/{channel}/sample.y.avg/value"])

                    # Process X channel: crop length -> reshape to physical pixels -> discard settling lag -> average
                    data_x = np.reshape(data_x_raw[:, :desired_len], [data_shape[0], px, -1])
                    data_x = data_x[:,:,settling_px:].mean(axis=2)
                    data[scan_idx, cdx, 0] = data_x
                    
                    # Process Y channel similarly
                    data_y = np.reshape(data_y_raw[:, :desired_len], [data_shape[0], px, -1])
                    data_y = data_y[:,:,settling_px:].mean(axis=2)
                    data[scan_idx, cdx, 1] = data_y

                scan_idx += 1

    return data, channels

# ==========================================
# Main Execution & Plotting Routine
# ==========================================

def main(folder_path, data, channels, align, align_side):
    """Load data, correct for drift, and plot data."""
    scan_data, ref_data = load_scan_files(folder_path)
    scan_shape = get_scan_shape(scan_data[0])
    topology_f_list, topology_b_list, shifts, masks, warp_success = [], [], [], [], []
    
    # Process reference image (Scan 0)
    ref_topo_f = fit_plane_and_shift(scan_data[0]['topo_f'].values.reshape(scan_shape))
    ref_topo_b = fit_plane_and_shift(scan_data[0]['topo_b'].values.reshape(scan_shape))
    ref_topo_f, _ = fit_poly_col_and_shift(ref_topo_f, align_side)
    ref_topo_b, _ = fit_poly_col_and_shift(ref_topo_b, align_side)

    topology_f_list.append(ref_topo_f)
    topology_b_list.append(ref_topo_b)
    masks.append(np.ones_like(ref_topo_f, dtype=np.uint8))
    warp_success.append(1)

    include_img_up_to = len(scan_data)
    
    # Process and align subsequent scans
    for scan in scan_data[1:include_img_up_to]:
        topo_f = scan['topo_f'].values.reshape(scan_shape)
        topo_b = scan['topo_b'].values.reshape(scan_shape)
        
        # Flatten topography
        topo_f = fit_plane_and_shift(topo_f)
        topo_b = fit_plane_and_shift(topo_b)
        topo_f, _ = fit_poly_col_and_shift(topo_f, align_side)
        topo_b, _ = fit_poly_col_and_shift(topo_b, align_side)

        if align:
            try:
                # Align to reference, save transformation matrices for the optical data
                aligned_topo_f, shift, mask = align_images(ref_topo_f, topo_f)
                aligned_topo_b, _, _ = align_images(ref_topo_b, topo_b)

                topology_f_list.append(aligned_topo_f)
                topology_b_list.append(aligned_topo_b)
                shifts.append(shift)
                masks.append(mask)
                warp_success.append(1)
            except cv2.error as e:
                print(e) # ECC failed to converge (e.g. sample moved entirely out of frame)
                warp_success.append(0)
        else:
            topology_f_list.append(topo_f)
            topology_b_list.append(topo_b)
            masks.append(np.ones_like(topo_f, dtype=np.uint8))
            warp_success.append(1)
    
    _ = np.sum(np.array(masks, dtype=np.float64), axis=0)
    avg_topo_f, _ = avg_w_mask(topology_f_list, masks)
    avg_topo_b, _ = avg_w_mask(topology_b_list, masks)

    # Feed calculated Topography shifts to align optical data
    h5_results = process_h5_file(data, channels, scan_shape, shifts, warp_success, align)
    print(f"Sucessful Aligns: {np.sum(warp_success)} of {len(warp_success)}")

    # Setup hardware-specific plot titles (demodulation harmonics in s-SNOM)
    titles = [[r"Cantilever",r"$38.9$ kHz"], 
              [r"Cantilever $\times$ Heat",r"$38.9$ kHz $-2 \cdot 1040.5$ Hz $=36.9$ kHz"],
              [r"Cantilever $\times$ Laser",r"$38.9$ kHz $-10$ kHz $=28.9$ kHz"],
              [r"Driving Freq",r"$1040.5$ Hz"],
              [r"Heating Freq",r"$2 \cdot 1040.5$ Hz $=2081$ Hz"],
              [r"Laser",r"$10$ kHz"]
            ]
    titles = [titles[int(j)] for j in channels]
    extent = [0, 7.1, 7.1, 0]  # Physical axes size (um)

    # Styling settings
    FONT_SIZE = 12
    FONT_SIZE_AXIS = FONT_SIZE + 2
    LABEL_PAD_AXIS = 1
    LABEL_PAD_AXIS_CB = 10
    FONT_SIZE_SUBTITLE = FONT_SIZE + 3
    FONT_SIZE_SUPTITLE = FONT_SIZE + 5
    plt.rcParams.update({'font.size': FONT_SIZE + 2})
    
    ROW_COL = 1  
    plot_layout = np.arange(1,9)
    plot_shape = [2,4]
    figure_size = (14, 8)
    shrink_cb = 0.6

    if ROW_COL:
        plot_shape = plot_shape[::-1]
        figure_size = figure_size[::-1]
        plot_layout = plot_layout.reshape(plot_shape).T.flatten()
        shrink_cb = 0.95

    percentile_clip = 75 # Cap color scale at upper quartile to prevent bright artifacts dominating

    # Plot an 8-panel figure per optical channel
    for idx, channel in enumerate(channels):
        plt.figure(figsize=figure_size)

        # Plot 1: Topography Forward
        ax1 = plt.subplot(*plot_shape, plot_layout[0])
        im1 = ax1.imshow(avg_topo_f*1e9, cmap='gray', extent=extent)  
        ax1.set_title('Mean Topology Forward', fontsize=FONT_SIZE_SUBTITLE)
        ax1.set_xlabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        ax1.set_ylabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        plt.colorbar(im1, ax=ax1, shrink=shrink_cb).set_label(label='Distance (nm)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 2: Optical Amplitude (R) Forward
        ax2 = plt.subplot(*plot_shape, plot_layout[1])
        im2 = ax2.imshow(np.clip(h5_results[channel]['R_f'],0,np.percentile(h5_results[channel]['R_f'], percentile_clip)), cmap='inferno', aspect=get_aspect(h5_results[channel]['R_f']), extent=extent)
        ax2.set_title(r'Mean $R$ Forward', fontsize=FONT_SIZE_SUBTITLE)
        ax2.set_xlabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        ax2.set_ylabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        plt.colorbar(im2, ax=ax2, shrink=shrink_cb).set_label(label='Intensity (a.u.)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 3: Optical Phase (Theta) Forward
        ax3 = plt.subplot(*plot_shape, plot_layout[2])
        im3 = ax3.imshow(h5_results[channel]['theta_f'], cmap='twilight', aspect=get_aspect(h5_results[channel]['theta_f']), vmin=-180, vmax=180, extent=extent)
        ax3.set_title(r'Mean $\theta$ Forward', fontsize=FONT_SIZE_SUBTITLE)
        ax3.set_xlabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        ax3.set_ylabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        plt.colorbar(im3, ax=ax3, shrink=shrink_cb).set_label(label='Phase (degrees)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 4: Optical Phase Standard Deviation Forward (maps sample edge noise)
        ax4 = plt.subplot(*plot_shape, plot_layout[3])
        im4 = ax4.imshow(np.clip(h5_results[channel]['theta_f_std'],0,np.percentile(h5_results[channel]['theta_f_std'], percentile_clip)), cmap=plt.get_cmap('inferno').reversed(), aspect=get_aspect(h5_results[channel]['theta_f_std']), extent=extent)
        ax4.set_title(r'SD $\theta$ Forward', fontsize=FONT_SIZE_SUBTITLE)
        ax4.set_xlabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        ax4.set_ylabel(r'Distance ($\mu$m)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS)
        plt.colorbar(im4, ax=ax4, shrink=shrink_cb).set_label(label='Phase (degrees)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # -- The same layout is repeated below for the Backward scan data --
        
        # Plot 5: Topography Backward
        ax5 = plt.subplot(*plot_shape, plot_layout[4])
        im5 = ax5.imshow(avg_topo_b*1e9, cmap='gray', extent=extent)
        ax5.set_title('Mean Topology Backward', fontsize=FONT_SIZE_SUBTITLE)
        # ... (axis labels omitted for brevity) ...
        plt.colorbar(im5, ax=ax5, shrink=shrink_cb).set_label(label='Distance (nm)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 6: Optical Amplitude (R) Backward
        ax6 = plt.subplot(*plot_shape, plot_layout[5])
        im6 = ax6.imshow(np.clip(h5_results[channel]['R_b'],0,np.percentile(h5_results[channel]['R_b'], percentile_clip)), cmap='inferno', aspect=get_aspect(h5_results[channel]['R_b']), extent=extent)
        ax6.set_title(r'Mean $R$ Backward', fontsize=FONT_SIZE_SUBTITLE)
        # ... (axis labels omitted for brevity) ...
        plt.colorbar(im6, ax=ax6, shrink=shrink_cb).set_label(label='Intensity (a.u.)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 7: Optical Phase (Theta) Backward
        ax7 = plt.subplot(*plot_shape, plot_layout[6])
        im7 = ax7.imshow(h5_results[channel]['theta_b'], cmap='twilight', aspect=get_aspect(h5_results[channel]['theta_b']), vmin=-180, vmax=180, extent=extent)
        ax7.set_title(r'Mean $\theta$ Backward', fontsize=FONT_SIZE_SUBTITLE)
        # ... (axis labels omitted for brevity) ...
        plt.colorbar(im7, ax=ax7, shrink=shrink_cb).set_label(label='Phase (degrees)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        # Plot 8: Optical Phase Standard Deviation Backward
        ax8 = plt.subplot(*plot_shape, plot_layout[7])
        im8 = ax8.imshow(np.clip(h5_results[channel]['theta_b_std'],0,np.percentile(h5_results[channel]['theta_b_std'], percentile_clip)), cmap=plt.get_cmap('inferno').reversed(), aspect=get_aspect(h5_results[channel]['theta_b_std']), extent=extent)
        ax8.set_title(r'SD $\theta$ Backward', fontsize=FONT_SIZE_SUBTITLE)
        # ... (axis labels omitted for brevity) ...
        plt.colorbar(im8, ax=ax8, shrink=shrink_cb).set_label(label='Phase (degrees)', fontsize=FONT_SIZE_AXIS, labelpad=LABEL_PAD_AXIS_CB)

        plt.suptitle(f's-SNOM Scan - {titles[idx][0]} - Channel {int(channel)+1}\nDemodulation Freq: {titles[idx][1]}', fontsize=FONT_SIZE_SUPTITLE)
        plt.tight_layout(pad=1.5) 
        plt.show()

# ==========================================
# Script Execution
# ==========================================

if __name__ == "__main__":
    # Define file paths
    main_folder_path = r"D:\Users\sSNOM\Documents\Zurich Instruments\LabOne\WebServer\session_20250829_161959_01"    
    scan_folder_path = r"Nanite_001"
    zurich_folder = r"Zurich_001_autosave_000"
    
    # Define parameters for sync and alignment
    align_side = "edges" # Use edges for scanner bow correction
    # Sync Params: [desired_x_pixels, AFM_line_time_s, Zurich_line_time_s, time_constant_s, filter_settling_multiplier]
    params = [32, 10, 11, 0.07115, 4] 
    
    # Run Loaders and Main Processing
    data, channels = load_h5_autosave_avg(os.path.join(main_folder_path, zurich_folder), params)
    main(os.path.join(main_folder_path, scan_folder_path), data, channels, align=1, align_side=align_side)