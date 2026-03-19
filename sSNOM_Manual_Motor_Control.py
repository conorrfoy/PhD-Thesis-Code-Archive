import sys
import time

# Add the custom module directory to the system path so Python can find it.
# sys.path[0] is the directory containing the script, so we insert at index 1.
sys.path.insert(1, 'D:/Users/sSNOM/Documents/aSNOM/Code Repo/Electronic_Modules')

# Import the custom LinearMotor control class from the local repository
from Koco_Linear_Actuator.linearmotor_comms import LinearMotor # type: ignore

if __name__=="__main__":
    # Brief pause to ensure system readiness before executing commands
    time.sleep(.5)
    
    # Define hardware IDs for the specific X, Y, and G (presumably Z, Gimbal, or Goniometer) motors
    motor_x_id = 842401042
    motor_y_id = 842400014
    motor_g_id = 842400276

    # Initialize the LinearMotor connection using a context manager (with statement)
    # This ensures the serial port is safely opened and automatically closed when done.
    with LinearMotor(serial_number="FT7AX4WAA") as lm:

        def move_motor_rel(motor_id, distance, backlash=20):
            """
            Moves the specified motor by a relative distance while compensating for mechanical backlash.
            
            It approaches the target position by first moving backward by the backlash amount, 
            and then moving forward by the total distance plus the backlash. This ensures the 
            gears are always engaged from the same direction.
            """
            # Move backward to clear backlash
            print(f"X Set Pos: {lm.move_relative(motor_id, distance=-backlash)}")
            print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id))}")
            
            # Move forward to the actual target distance (compensating for the backward move)
            print(f"X Set Pos: {lm.move_relative(motor_id, distance=distance+backlash)}")
            print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id))}")


        def move_motor_abs(motor_id, distance, backlash=20):
            """
            Moves the specified motor to an absolute distance/position while compensating for backlash.
            
            Similar to the relative move, it first goes to an offset position (-backlash) 
            and then drives forward to the final absolute position (+backlash).
            """
            # Move to an absolute position offset by the negative backlash amount
            print(f"X Set Pos: {lm.move_absolute(motor_id, distance=-backlash)}")
            print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id))}")
            
            # Move to the final absolute position, adding the backlash back
            print(f"X Set Pos: {lm.move_absolute(motor_id, distance=distance+backlash)}")
            print(f"X Cur Pos: {lm.steps2micron(lm.get_position(motor_id))}")


        # ##### Home Motors #####
        print(f"X HomePos: {lm.steps2micron(lm.home_motor(id=motor_x_id))}")
        print(f"Y HomePos: {lm.steps2micron(lm.home_motor(id=motor_y_id))}")
        print(f"G HomePos: {lm.steps2micron(lm.home_motor(id=motor_g_id))}")
        print()

        # ##### Move Relative #####
        # Command each motor to move relatively (with standard backlash compensation)
        move_motor_rel(motor_x_id, 1) # (um)
        move_motor_rel(motor_y_id, 1) # (um)
        move_motor_rel(motor_g_id, 1) # (um)
        print() # Print empty line for terminal readability

        # ##### Move Absolute #####
        # Command each motor to move to absolute position from Home (with standard backlash compensation)
        move_motor_abs(motor_x_id, 1) # (um)
        move_motor_abs(motor_y_id, 1) # (um)
        move_motor_abs(motor_g_id, 1) # (um)
        print()