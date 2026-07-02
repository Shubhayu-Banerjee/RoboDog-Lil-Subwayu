import math
import numpy as np

# Constants
LS1 = 8.0
LS2 = 12.0
MIN_EXT = abs(LS1 - LS2) + 0.01
MAX_EXT = LS1 + LS2 - 0.01
LS1_ANGLE_MIN = 210.0
LS1_ANGLE_MAX = 30.0
BETA_MIN = 20.0
BETA_MAX = 150.0
ROLL_MIN = -30.0
ROLL_MAX = 90.0


def solve_dog_leg(target_x, target_y, target_z):
    """
    Solves IK for the dog leg.
    Returns: [Pitch_Angle, Roll_Angle, Extension, Alpha, Beta]
    or ['OFB'] if inaccessible.
    """

    # 1. Roll calculation (Abduction/Adduction)
    # The leg rotates around the X-axis (forward/back).
    # Roll is the angle between the Z-axis and the YZ plane projection.
    if target_y == 0 and target_z == 0:
        roll = 0.0
    else:
        roll = math.degrees(math.atan2(target_y, -target_z))

    if not (ROLL_MIN <= roll <= ROLL_MAX):
        return ['OFB']

    # 2. Local 2D Projection
    # Project 3D point into the local 2D plane (X, Z_local)
    z_loc = -math.sqrt(target_y ** 2 + target_z ** 2)
    x_loc = target_x

    extension = math.hypot(x_loc, z_loc)
    if not (MIN_EXT <= extension <= MAX_EXT):
        return ['OFB']

    # 3. Geometric IK (Triangle Solver)
    # Law of Cosines for Beta (Angle at C)
    cos_beta = (LS1 ** 2 + LS2 ** 2 - extension ** 2) / (2 * LS1 * LS2)
    beta = math.degrees(math.acos(np.clip(cos_beta, -1.0, 1.0)))

    if not (BETA_MIN <= beta <= BETA_MAX):
        return ['OFB']

    # Law of Cosines for Alpha (Angle at A)
    cos_alpha = (LS1 ** 2 + extension ** 2 - LS2 ** 2) / (2 * LS1 * extension)
    alpha = math.degrees(math.acos(np.clip(cos_alpha, -1.0, 1.0)))

    # Pitch Angle (Absolute angle of the leg in the local plane)
    pitch = math.degrees(math.atan2(z_loc, x_loc)) + 90

    # 4. Constraint Check (LS1 Angle)
    # For a dog leg, we assume a standard elbow-back configuration
    # The absolute angle of LS1 link:
    # A = (0,0), C = (LS1 * cos(theta), LS1 * sin(theta))
    # Note: Simplified for a standard kinematic branch
    ls1_dir = math.degrees(math.atan2(z_loc, x_loc))  # Simplified vector

    # Normalize for constraint check
    phi = ls1_dir % 360.0

    # Check if inside allowed cone [210, 330] or [330, 30] logic
    if not ((210.0 <= phi <= 360.0) or (0.0 <= phi <= 30.0)):
        return ['OFB']

    return [round(alpha, 2), round(beta, 2), round(roll, 2)]

# Example Usage:
result = solve_dog_leg(9.4, 7.4, -10.8)
print(result)
