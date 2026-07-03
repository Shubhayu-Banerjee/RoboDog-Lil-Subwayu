# Quadruped Robot Simulator

<img width="1012" height="792" alt="image" src="https://github.com/user-attachments/assets/131b0561-8a97-4138-a94e-0448f07461a7" />


A Python-based quadruped robot simulator featuring inverse kinematics, static stability analysis, reactive stepping, and continuous gait generation.



## Features



-  Two-link inverse kinematics for each leg
-  Static stability monitoring using the support polygon
-  Reactive stepping when a leg reaches its workspace limit
-  Continuous walking using a CPG-inspired gait
-  Interactive 3D visualization with Matplotlib
-  GUI controls for body movement and stance adjustment
-  Adjustable chassis height
-  Joint and roll constraint checking


---



## Robot Model



Each leg consists of:



- Hip

- Knee

- Foot



with two rigid links:



- Upper Leg (`LS1`)

- Lower Leg (`LS2`)



The robot body is modeled as a rectangular chassis with four corner-mounted legs.



```

      Front



 FL -------- FR

 |            |

 |  Chassis   |

 |            |

 RL -------- RR



      Rear

```



---



## Main Components



### Inverse Kinematics



The solver computes the knee position for every leg while respecting:



- Reachability constraints

- Knee angle limits

- Hip joint limits

- Roll limits



---



### Workspace Planner



Each leg has a circular workspace.



Whenever a foot approaches the workspace boundary, the planner computes a new target and initiates a step.



---



### Reactive Stepping



The robot:



1. Moves the body.

2. Checks every leg.

3. Detects workspace violations.

4. Repositions affected legs.

5. Restores balance.



---



### Continuous Walking



The merged version also includes a continuous walking mode based on a CPG-style gait generator.



Instead of waiting for workspace violations, the robot repeatedly generates coordinated footsteps to produce smooth locomotion.



---



### Stability Controller



Before lifting a leg, the simulator verifies that the robot's center of gravity remains inside the support polygon formed by the remaining feet.



If instability is detected, the controller:



- shifts the body toward the support polygon centroid,

- performs the step,

- returns the body to its desired position.



---



### Visualization



The renderer displays:



- Chassis

- Legs

- Knees

- Feet

- Support polygon

- Center of gravity

- Workspace boundaries

- Stability warnings

- Joint-limit violations



---



## Controls

| Control | Function |
|----------|----------|
| Height Slider | Adjust chassis height |
| Forward | Move body forward |
| Back | Move body backward |
| Left | Move body left |
| Right | Move body right |
| Stance | Reset the robot to a stable stance |
| Walk | Enable continuous walking mode |



---



## Technologies Used



- Python

- NumPy

- Matplotlib

- SciPy



---



## Future Improvements



- Full 3D inverse kinematics

- Terrain adaptation

- ROS2 integration

- PyBullet / MuJoCo simulation

- Dynamic stability (ZMP/MPC)

- Hardware deployment



---



## Project Overview



```

Configuration

│
├── Robot Geometry
├── Joint Limits
├── Workspace Parameters
│

Inverse Kinematics

│
├── Reachability
├── Knee Solver
├── Roll Constraints
│

Motion Planning

│
├── Reactive Stepping
├── Continuous Walking
├── Stability Correction
│

Visualization

│
├── Chassis
├── Legs
├── Support Polygon
└── GUI Controls

```



## Summary



This project demonstrates the core principles of quadruped locomotion by combining analytical inverse kinematics, workspace-aware footstep planning, static stability analysis, and continuous gait generation into a single interactive simulator.
