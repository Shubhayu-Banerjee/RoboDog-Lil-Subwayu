# 4-LEG Inverse Kinematics

<img width="356" height="369" alt="image" src="https://github.com/user-attachments/assets/899e3933-3d74-4b33-a9a0-5274e9ae9de4" />

A Python-based **quadruped robot simulator** that demonstrates:

* 3D inverse kinematics
* Static stability checking
* Automatic footstep planning
* Compensatory body sway
* Interactive visualization using Matplotlib

The simulator models a four-legged robot capable of maintaining balance while moving its body and repositioning its feet inside predefined workspaces.

---

# Features

*  3D quadruped visualization
*  Two-link inverse kinematics solver
*  Roll-angle constraints
*  Joint angle constraints
*  Static stability analysis
*  Convex support polygon visualization
*  Automatic stepping gait
*  Compensatory center-of-gravity shifting
*  Interactive GUI controls

---

# Robot Model

Each leg consists of:

```
Hip
 │
 │  (Upper Leg)
 │
Knee
 │
 │  (Lower Leg)
 │
Foot
```

The robot body is represented as a rectangular chassis.

Each leg is attached to one corner of the chassis:

```
      Front

   FL ------- FR
    |         |
    | Chassis |
    |         |
   RL ------- RR

       Rear
```

---

# Simulation Components

## 1. Chassis

The chassis stores:

* Position
* Height
* Geometry

```python
body = {
    "x": ...,
    "y": ...,
    "z": ...
}
```

The body moves independently while the legs automatically reposition themselves whenever they approach their workspace limits.

---

## 2. Foot Positions

Each foot is stored globally.

```python
foot_positions = {
    "FR": ...,
    "FL": ...,
    "RR": ...,
    "RL": ...
}
```

These positions are updated whenever a stepping animation occurs.

---

## 3. Hip Positions

Hip positions are computed from the body position.

```
Body Position
      ↓
Corner Offset
      ↓
Hip Position
```

This is handled by:

```python
get_hip_positions()
```

---

# Inverse Kinematics

Each leg is modeled as a **2-link planar manipulator**.

```
      Hip
       ●
      / \
 LS1 /   \
    /     \
   ●-------●
 Knee     Foot
      LS2
```

The solver computes the knee position using the law of cosines.

The algorithm:

1. Measure hip-foot distance
2. Check reachability
3. Compute knee angle
4. Test both elbow configurations
5. Apply joint limits
6. Return the valid solution

Implemented in:

```python
solve_leg_local_ik()
```

---

# Joint Constraints

The simulation enforces realistic robot limits.

## Upper leg

```
Allowed:
210° → 30°
```

Configured using:

```python
LS1_ANGLE_MIN
LS1_ANGLE_MAX
```

---

## Knee

```
20° ≤ β ≤ 150°
```

Configured using:

```python
BETA_MIN
BETA_MAX
```

---

## Roll

Each leg also has roll constraints.

```
ROLL_MIN = -45°
ROLL_MAX = 45°
```

Any violation causes the leg to be marked invalid.

---

# Workspace

Every leg owns a circular workspace.

```
        ○
    ○       ○

  ○     Hip     ○

    ○       ○
        ○
```

Radius:

```python
WORKSPACE_TRIGGER_RADIUS
```

When the foot approaches the boundary, the planner schedules a new step.

---

# Step Planning

Instead of stepping randomly, the simulator computes an ideal target.

```
Current Foot
      ●

          →

      Target
```

The target is generated using

```python
compute_step_target()
```

which places the foot toward the edge of its workspace in the outward direction from the robot's center of gravity.

---

# Walking Algorithm

Movement follows this sequence:

```
Move Body

      ↓

Check Every Leg

      ↓

Outside Workspace?

      ↓

Yes

      ↓

Maintain Stability

      ↓

Move Foot

      ↓

Continue
```

Implemented in:

```python
animate_leg_steps()
```

---

# Stability Analysis

Whenever a foot lifts, only three feet remain on the ground.

Those three feet form the support triangle.

```
Foot ●
      |\
      | \
      |  \
      |   \
      ●----●
```

The simulator determines whether the center of gravity lies inside this triangle.

This is computed by:

```python
get_triangle_margin()
```

Positive value:

```
COG inside triangle
```

Negative value:

```
Robot unstable
```

---

# Compensatory Body Sway

If the robot becomes unstable while lifting a leg:

1. Compute support triangle centroid
2. Shift body toward centroid
3. Execute the step
4. Return body to its intended position

This mimics how real quadruped robots redistribute weight before stepping.

---

# Automatic Pre-positioning

If shifting the body is insufficient, the simulator searches for a better helper-leg placement.

The algorithm:

* Choose another leg
* Sample multiple candidate positions around its workspace
* Simulate each placement
* Evaluate the resulting stability
* Select the safest option

Implemented in:

```python
try_pre_position_leg()
```

This proactive strategy reduces the chance of entering unstable configurations.

---

# Stepping Animation

A step follows a smooth trajectory.

The foot follows a cosine-interpolated path while lifting vertically using a sine curve.

Implemented in:

```python
step_leg()
```

---

# Rendering

Visualization is handled entirely with Matplotlib.

The renderer displays:

* Chassis
* Legs
* Knees
* Feet
* Workspace circles
* Support polygon
* Center of gravity
* Stability warnings
* Joint-limit warnings

Main renderer:

```python
draw()
```

---

# User Controls

## Height Slider

Adjusts robot body height.

```
Lower
───────────────►
Higher
```

---

## Direction Buttons

```
Forward

Left   Stance   Right

Backward
```

Buttons move the body while the gait planner automatically repositions the legs.

---

# Code Structure

```
Configuration
│
├── Robot geometry
├── Joint limits
├── Workspace limits
│
Kinematics
│
├── IK Solver
├── Angle helpers
├── Geometry
│
Motion Planning
│
├── Step target generation
├── Body shifting
├── Step animation
├── Stability correction
│
Renderer
│
├── Chassis
├── Legs
├── Support polygon
├── Warnings
│
GUI
│
├── Slider
├── Buttons
└── Event callbacks
```

---

# Technologies Used

* Python
* NumPy
* Matplotlib
* SciPy (`ConvexHull`)

---

# Future Improvements

Potential extensions include:

* Full 3D inverse kinematics with hip yaw
* Dynamic gait generation (Trot, Pace, Bound)
* Terrain adaptation
* Collision detection
* Physics engine integration (PyBullet or MuJoCo)
* Real-time joystick/gamepad control
* ROS 2 integration
* Energy-efficient footstep planning
* Dynamic stability using Zero Moment Point (ZMP) or Model Predictive Control (MPC)

---

# Summary

This project demonstrates the complete pipeline of a statically stable quadruped walking controller:

* Analytical inverse kinematics
* Workspace-aware footstep planning
* Static stability verification
* Automatic body compensation
* Smooth stepping animation
* Interactive 3D visualization

It serves as an educational simulation for understanding quadruped locomotion and robot kinematics before integrating with a physical robot or a full physics simulator.
