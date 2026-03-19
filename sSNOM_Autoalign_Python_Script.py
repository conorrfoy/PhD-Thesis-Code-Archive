import sys, os
# Add the custom module directory to the system path so Python can find it.
# sys.path[0] is the directory containing the script, so we insert at index 1.
sys.path.insert(1, 'D:/Users/sSNOM/Documents/aSNOM/Code Repo/Electronic_Modules')

# import file
from Koco_Linear_Actuator.linearmotor_comms import LinearMotor # type: ignore
from Zurich_Align_Avg import get_scan_shape, fit_plane_and_shift, fit_cubic_col_and_shift, align_images # type: ignore
import numpy as np
import pandas as pd
import time, glob, re
import cv2

# ==========================================
# Data Loading & Verification
# ==========================================

def load_ref_and_newest_files(folder_path):
    """
    Loads the first (reference) scan and the most recently completed scan.
    Includes a safety check to ensure the trigger file points to the expected data.
    """
    # Grab all scan files and sort them numerically
    scan_files = sorted(glob.glob(os.path.join(folder_path, "scan_*.csv")), key=lambda x: int(re.search(r'scan_(\d+)_', x).group(1)))
    
    # Read the align_flag.txt to see which file triggered the alignment request
    with open(os.path.join(folder_path, "align_flag.txt"), 'r') as file:
        data_path = file.read().rstrip()
        data_path_rel = data_path.split(os.path.sep)[1:]  # Drop the drive letter
        scan_file_rel = scan_files[-1].split(os.path.sep)[-len(data_path_rel):]
    
    # Safety Check: Ensure the file that triggered the flag is actually the newest file we found
    if os.path.join(*data_path_rel) != os.path.join(*scan_file_rel):
        raise Exception("Mismatch in data file")
    
    # We need at least 2 scans to calculate drift (a reference and a new target)
    if len(scan_files) < 2:
        return None  

    # Load the metadata (from scan 0), reference data (scan 0), and new data (latest scan)
    df_consts = pd.read_csv(scan_files[0], nrows=1, header=None, names=["image_width", "image_height", "time_line", "points_line", "lines", "rotation"])
    df_ref = pd.read_csv(scan_files[0], skiprows=1, header=None, names=["x", "y", "topo_f", "topo_b", "amp_f", "amp_b"])
    df_new = pd.read_csv(scan_files[-1], skiprows=1, header=None, names=["x", "y", "topo_f", "topo_b", "amp_f", "amp_b"])
    
    return df_ref, df_new, df_consts

# ==========================================
# Core Alignment & Motor Control Logic
# ==========================================

def main(folder_path, motor_id, num_scans):
    """
    Checks for an alignment request, calculates physical drift between scans, 
    and drives the motors to correct it.
    """
    align_flag = os.path.join(folder_path, "align_flag.txt")
    
    # Only execute if the external acquisition software has created the align flag
    if os.path.isfile(align_flag):
        scan_data = load_ref_and_newest_files(folder_path)
        
        if scan_data != None:
            scan_shape = get_scan_shape(scan_data[0])
            
            # Prepare Reference Image: Flatten and remove scanner bow
            ref_topo_f = fit_plane_and_shift(scan_data[0]['topo_f'].values.reshape(scan_shape))
            ref_topo_f, _ = fit_cubic_col_and_shift(ref_topo_f, 'left')

            # Prepare Newest Image: Flatten and remove scanner bow
            topo_f = fit_plane_and_shift(scan_data[1]['topo_f'].values.reshape(scan_shape))
            topo_f, _ = fit_cubic_col_and_shift(topo_f, 'left')

            try:
                # Calculate pixel shift between reference and new scan
                _, shift, _ = align_images(ref_topo_f, topo_f)

                # Extract physical scan dimensions to convert pixel shifts to real-world distances
                img_height, img_width, img_rotation = scan_data[2][["image_height", "image_width", "rotation"]].values[0]
                rot = [np.cos(np.deg2rad(img_rotation)), np.sin(np.deg2rad(img_rotation))]
                
                # Convert relative pixel shift (0 to 1 scale) into micrometers
                x_img_mov = shift[0, 2] * img_width * 1e6
                y_img_mov = shift[1, 2] * img_height * 1e6
                
                # Coordinate Transform: Map the image's coordinate system to the motor's physical axes
                x_motor_mov = - y_img_mov * rot[0] - x_img_mov * rot[1]
                y_motor_mov = x_img_mov * rot[0] - y_img_mov * rot[1]

                # Connect to hardware and execute the correction move
                with LinearMotor(serial_number="FT7AX4WAA") as lm:

                    # ##### Move Relative #####
                    print([x_motor_mov, y_motor_mov])
                    
                    # X Motor Move (with 20um backlash compensation)
                    print(f"X Set Pos: {lm.move_relative(motor_id[0], distance=-20)}")
                    print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id[0]))}")
                    print(f"X Set Pos: {lm.move_relative(motor_id[0], distance=20+x_motor_mov)}")
                    print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id[0]))}")
                    
                    # Y Motor Move (with 20um backlash compensation)
                    print(f"Y Set Pos: {lm.move_relative(motor_id[1], distance=-20)}")
                    print(f"Y Cur Pos: {lm.steps2micron(lm.get_position(motor_id[1]))}")
                    print(f"Y Set Pos: {lm.move_relative(motor_id[1], distance=20+y_motor_mov)}")
                    print(f"Y Cur Pos: {lm.steps2micron(lm.get_position(motor_id[1]))}")

            except cv2.error as e:
                # Catch instances where the image moved too far to correlate
                print(e)

        # Remove the flag so next scan can begin
        os.remove(align_flag)
        print(f"Scan {num_scans} complete")
        num_scans -= 1
        
        # Once all scans are complete remove the Stop flag to trigger end of scanning
        if not num_scans:
            os.remove(os.path.join(folder_path, "stop_flag.txt"))
            
    return num_scans


# ==========================================
# Main Execution Loop
# ==========================================

if __name__=="__main__":
    folder_path = r"D:\Users\sSNOM\Documents\Zurich Instruments\LabOne\WebServer\session_20250925_101048_06\Nanite_000"
    
    motor_id = [842401042, 842400014] # x, y
    num_scans = 25

    # Continuous Polling Loop:
    # As long as the 'stop_flag.txt' exists in the directory, this script will keep checking for 'align_flag.txt' every 1 second.
    while os.path.isfile(os.path.join(folder_path, "stop_flag.txt")):
        num_scans = main(folder_path, motor_id, num_scans)
        time.sleep(1)