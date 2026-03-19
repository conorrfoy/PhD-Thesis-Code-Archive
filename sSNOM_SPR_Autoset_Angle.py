import sys
# Add the custom module directory to the system path so Python can find it.
# sys.path[0] is the directory containing the script, so we insert at index 1.
sys.path.insert(1, 'D:/Users/sSNOM/Documents/aSNOM/Code Repo/Electronic_Modules')

# Import custom hardware control modules and standard libraries
from Koco_Linear_Actuator.linearmotor_comms import LinearMotor # type: ignore
from sSNOM_Photodiode_and_IRLED.sSNOM_Photodiode_and_IRLED import ADC # type: ignore
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from datetime import datetime

def microns_to_deg(position):
    """
    Converts linear motor travel (microns) to an angle (degrees) based on a calibrated linear equation.
    Formula: Degrees = -(steps / 1240) + (3965 / 62)
    """
    return -(position / 1240) + 3965/62

if __name__=="__main__":
    # Log the start time of the scan
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    motor_g_id = 842400276 # ID for the Goniometer/Rotational motor

    # Define the scan parameters: start, stop, and step size (all in microns of linear motor travel)
    scan = np.arange(18000, 25000, 300) 
    avg_reps = 5 # Number of ADC readings to average at each position to reduce noise

    # Initialize a 2D array to hold the trace data: [position, adc_intensity]
    trace = np.zeros([np.shape(scan)[0], 2], dtype=np.float32)

    # Open connections to the motor and the ADC using context managers
    with LinearMotor(serial_number="FT7AX4WAA") as lm:
        def move_abs(pos, verbose=0, g_id=842400276):
            """
            Moves the specified motor to an absolute motor position (microns).
            """
            pos = lm.move_absolute(g_id, pos) # Command motor to move to 'pos' (in microns)
            if verbose: print(f"G Set Pos: {pos}")
            
            # Read the actual position and convert it from steps to microns
            cur_pos = lm.steps2micron(lm.get_position(g_id)) 
            if verbose: print(f"G Cur Pos: {cur_pos}")
            
            return cur_pos # Returns position in microns

        with ADC(location="1-8.1.4") as adc: # Location is unique for each system
            
            # Disable continuous read to allow for manual, synchronized triggering
            adc.disable_continous_ADC_read()

            # ##### 1. Data Acquisition Loop #####
            for i, pos in enumerate(tqdm(scan)):
                
                # Move the motor and record the actual returned position (in microns)
                trace[i,0] = move_abs(pos) 
                
                # Take multiple ADC readings and average them
                for _ in range(avg_reps):
                    trace[i,1] += adc.read_ADC()
                trace[i,1] /= avg_reps

            # ##### 2. Data Processing & Optimization #####
            # Calculate the derivative (difference between consecutive intensity points)
            # Then scale the resulting derivative to match the min/max of the original trace for plotting
            trace_diff = minmax_scale(np.diff(trace[:,1]), (trace[:,1].min(), trace[:,1].max()))
            
            # Find the index where the rate of change is highest (steepest slope of the SPR curve)
            opt_idx = np.argmax(trace_diff)
            
            # Calculate the optimal step position by averaging the two steps that bound the highest derivative
            opt_pos = np.round(np.mean([trace[opt_idx, 0], trace[opt_idx+1, 0]]))
            
            # Move the motor to the newly found optimal position
            move_abs(opt_pos, verbose=1)
            
            # Take an averaged ADC reading at this optimal position to display on the plot
            opt_value = 0.0
            for _ in range(avg_reps):
                opt_value += adc.read_ADC()
            opt_value /= avg_reps
            
            # Log completion time and optimal position
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f"opt pos:{opt_pos}")

    # ##### 3. Data Visualization #####
    plt.xlabel('Angle [degrees]') 
    plt.ylabel('intensity [rel. units]')
    plt.title(f'Autoset SPR Angle (Pos: {opt_pos}, {microns_to_deg(opt_pos)} deg)')

    # Plot the main SPR intensity curve. 
    plt.plot(microns_to_deg(trace[:,0]), trace[:,1], label="SPR")
    
    # Plot the derivative. The x-axis is shifted by half a step to align with the midpoint of the difference.
    plt.plot(microns_to_deg(trace[:-1,0]+(scan[1]-scan[0])/2), trace_diff, label="Diff. SPR")
    
    # Plot vertical and horizontal lines to highlight the optimal point
    plt.axvline(microns_to_deg(opt_pos), label="Opt. Angle")
    plt.axhline(opt_value)
    
    plt.legend()
    plt.show()