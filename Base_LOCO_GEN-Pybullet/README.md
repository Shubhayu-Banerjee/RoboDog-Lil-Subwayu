# Robo Dog — Contact-Aware CPG Controller

A PyBullet simulation controller for a quadruped robot. The project combines a diagonal Central Pattern Generator (CPG), analytical inverse kinematics (IK), binary foot-contact feedback, keyboard control, body-pose sliders, and a separate actuator telemetry dashboard.

The controller is designed around **binary contact information**: a foot is either in contact with the terrain or it is not. It does not require pressure sensors or force measurements for its contact-feedback logic.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
  - [Diagonal CPG](#diagonal-cpg)
  - [Foot Trajectory](#foot-trajectory)
  - [Contact-Aware Feedback](#contact-aware-feedback)
  - [Analytical Inverse Kinematics](#analytical-inverse-kinematics)
- [Requirements](#requirements)
- [Project Files](#project-files)
- [Running the Simulation](#running-the-simulation)
- [Controls](#controls)
- [PyBullet GUI Sliders](#pybullet-gui-sliders)
- [Configuration and Tuning](#configuration-and-tuning)
- [Foot Contact Switches: Simulation and Hardware](#foot-contact-switches-simulation-and-hardware)
- [Actuator Telemetry](#actuator-telemetry)
- [Important Implementation Notes](#important-implementation-notes)
- [Troubleshooting](#troubleshooting)
- [Safety Notes](#safety-notes)

---

## Features

- **Diagonal CPG gait:** coordinates diagonal leg pairs using a shared oscillator.
- **Contact-aware leg timing:** each leg receives a local phase correction based on its own binary contact state.
- **Early-touchdown response:** a foot that contacts terrain during swing can transition toward stance early.
- **Missed-touchdown search:** a leg that has not found the ground near the end of swing slows locally and lowers gradually.
- **Stance contact-loss response:** a leg that unexpectedly loses contact during stance is held back in phase rather than immediately being lifted.
- **Analytical IK:** calculates hip-roll, thigh, and knee targets from a desired foot position.
- **Smooth command transitions:** velocity commands ramp instead of changing instantaneously; stopping transitions through a controlled gait state.
- **Body-pose controls:** interactive sliders for body height, roll, pitch, and static yaw.
- **Stride scaling:** separate X and Y stride sliders.
- **Jump state machine:** a scripted jump sequence triggered by the spacebar.
- **Actuator telemetry dashboard:** displays estimated total servo current, a current status band, instantaneous simulated horizontal speed, and a rolling 5-second speed average.
- **Asynchronous telemetry:** torque/current calculations and plotting are kept off the main PyBullet control path.

---

## How It Works

### Diagonal CPG

The controller runs a global oscillator at a configured frequency. The four nominal leg phases are:

```python
nominal_phases = [
    phase,
    phase + math.pi,
    phase + math.pi,
    phase
]
```

This coordinates the legs as two diagonal pairs. Contact feedback adds a bounded, per-leg phase correction. The global oscillator continues running even when an individual leg needs to recover.

The configured nominal gait frequency is **2.0 Hz**.

### Foot Trajectory

The `_foot_trajectory()` function maps each leg's phase to a normalized horizontal position and vertical lift:

- **Stance:** the foot travels backward from normalized X `+1` to `-1`, with zero nominal lift.
- **Swing:** the foot travels forward from `-1` to `+1`, while a smooth sinusoidal profile raises and lowers it.

The normalized trajectory is scaled by the commanded X and Y stride. Swing height is also ramped smoothly when movement starts or stops.

### Contact-Aware Feedback

The simulated contact-switch reader checks PyBullet collision contacts on the configured final leg links. It accepts contact with the ground plane and the generated ramps, and uses a contact-normal direction check to reject side collisions. A short debounce prevents one-frame contact changes from immediately changing the accepted switch state.

The feedback layer applies local corrections:

1. **Early contact during swing:** moves the local phase toward the stance boundary.
2. **Late or missed touchdown:** near the end of swing, reduces local phase and gradually lowers the leg in search of contact.
3. **Contact lost during stance:** biases the local phase backward to avoid immediately lifting the leg.
4. **Contact restored:** allows the accumulated correction/search offset to decay.

These corrections are bounded. The design intent is to let one leg recover without freezing the entire oscillator.

### Analytical Inverse Kinematics

The controller uses two configurable leg lengths:

- Thigh: `0.15 m`
- Calf: `0.20 m`

For each leg, the controller calculates three joint targets from the requested foot position. The cosine terms used by the law-of-cosines calculations are clamped to valid ranges before inverse trigonometric functions are evaluated.

The code assumes the robot's URDF joint ordering matches the controller's 12-joint layout: three joints per leg, in the same leg order used by the controller.

---

## Requirements

Install the Python dependencies in the Python environment you intend to use:

```bash
python -m pip install numpy pybullet matplotlib
```

The script also uses Python standard-library modules:

- `time`
- `math`
- `multiprocessing`
- `threading`
- `collections.deque`

Use a Python version supported by the installed versions of PyBullet, NumPy, and Matplotlib.

---

## Project Files

At minimum, keep the controller and robot description together:

```text
your-project/
├── Base_LOCO_GEN-Pybullet.py   # Controller script (your filename may differ)
└── robot.urdf                  # Robot model loaded by the controller
```

The simulation also loads PyBullet's built-in `plane.urdf` through `pybullet_data`.

**Important:** the controller calls `pyb.loadURDF("robot.urdf", ...)`. Run it from a working directory where `robot.urdf` can be found, or change that path to the correct location. Meshes and other assets referenced by the URDF must also be accessible.

---

## Running the Simulation

1. Confirm that `robot.urdf` is the intended model and that its joint order matches the controller.
2. Install the required packages.
3. Open a terminal in the project directory.
4. Run the controller:

   ```bash
   python Base_LOCO_GEN-Pybullet.py
   ```

5. Use the PyBullet window for locomotion and pose controls. A separate Matplotlib window displays actuator telemetry.
6. Stop the controller with `Ctrl+C` in the terminal or close the relevant windows.

On Windows, the script calls `multiprocessing.freeze_support()` in its entry point. Keep the usual `if __name__ == "__main__":` guard if you reorganize the multiprocessing code.

---

## Controls

The controller prints its controls at startup. The mappings in the current source are:

| Key | Action |
|---|---|
| `K` | Forward (controller's `-X` direction) |
| `N` | Backward (controller's `+X` direction) |
| `B` | Strafe left |
| `M` | Strafe right |
| `J` | Rotate counter-clockwise |
| `L` | Rotate clockwise |
| `X` | Request a smooth stop |
| `Space` | Trigger the scripted jump sequence |
| `Arrow keys` | Adjust the PyBullet camera |

The forward/backward, strafe, and yaw command channels are sign-inverted at the command-to-foot-trajectory mapping layer to preserve the intended user-facing directions with the stance/swing trajectory convention. The contact-feedback and CPG phase logic are not inverted.

Movement commands ramp toward their requested values. Releasing the movement keys allows the command values to decay; pressing `X` explicitly requests zero command and initiates the smooth stopping process.

---

## PyBullet GUI Sliders

| Slider | Range | Default | Purpose |
|---|---:|---:|---|
| Body Height | `-0.30` to `-0.05` | `-0.17` | Adjusts the nominal vertical foot target/body stance parameter |
| Body Roll | `-0.4` to `0.4` | `0.0` | Applies a roll pose |
| Body Pitch | `-0.4` to `0.4` | `0.0` | Applies a pitch pose |
| Static Body Yaw | `-0.5` to `0.5` | `0.0` | Applies a static yaw pose |
| X Stride | `1.0` to `2.0` | `1.0` | Scales the X stride |
| Y Stride | `1.0` to `2.0` | `1.0` | Scales the Y stride |

The X and Y stride sliders scale the corresponding command-driven foot displacement. Increasing stride can increase speed, but it also increases the distance the legs must cover and may reduce stability depending on the model, friction, actuator limits, and terrain.

---

## Configuration and Tuning

The principal configuration values are defined near the top of the Python file.

### Leg geometry and nominal stance

```python
LEG_THIGH_LENGTH = 0.15
LEG_CALF_LENGTH = 0.20
NEUTRAL_KNEE_ANGLE = 1.00
STANCE_FRACTION = 0.60
NOMINAL_FOOT_X = 0.12
```

- Change the thigh and calf lengths to match the modeled leg geometry.
- `STANCE_FRACTION` is the fraction of a cycle assigned to stance; the remainder is swing.
- `NOMINAL_FOOT_X` is the nominal horizontal reach parameter.

Changing the geometry constants does not automatically modify the physical dimensions in `robot.urdf`. Keep the controller's geometry and URDF geometry consistent.

### Command and gait transitions

```python
COMMAND_RAMP_TIME = 0.15
STOP_RAMP_TIME = 0.20
STEP_HEIGHT_RISE_TIME = 0.12
STEP_HEIGHT_FALL_TIME = 0.16
MAX_STEP_HEIGHT = 0.05
COMMAND_DEADZONE = 0.008
```

- Increase command ramp time for gentler command changes.
- Increase stop ramp time for a slower transition to standing.
- Increase step-height ramp times to make the swing-height transition more gradual.
- `MAX_STEP_HEIGHT` is the nominal maximum swing lift, in metres.

### Contact feedback

```python
CONTACT_DEBOUNCE_FRAMES = 2
MAX_PHASE_CORRECTION = 0.55
PHASE_CORRECTION_RATE = 8.0
CONTACT_SEARCH_START = 0.78
MAX_CONTACT_SEARCH_Z = 0.025
CONTACT_SEARCH_RATE = 0.10
STANCE_LOST_CONTACT_HOLD = 0.40
```

- `CONTACT_DEBOUNCE_FRAMES`: number of consecutive controller frames required to accept a changed switch state.
- `MAX_PHASE_CORRECTION`: maximum magnitude of an individual leg's phase correction, in radians.
- `PHASE_CORRECTION_RATE`: limits how quickly the local phase correction can change.
- `CONTACT_SEARCH_START`: normalized swing progress at which the missed-touchdown search begins.
- `MAX_CONTACT_SEARCH_Z`: maximum downward search offset, in metres.
- `CONTACT_SEARCH_RATE`: rate at which the search offset increases downward, in metres per second.
- `STANCE_LOST_CONTACT_HOLD`: phase-bias rate used when expected stance contact is absent.

Tune these values incrementally. Too much correction can make the local leg timing diverge from the global rhythm; too little may fail to recover a missed touchdown promptly.

### Actuator model and telemetry

```python
ACTUATOR_MAX_TORQUE = 2.45
ACTUATOR_MAX_CURRENT = 3.70
SERVO_SUPPLY_VOLTAGE = 6.0
```

These values feed an **estimate**, not a measured electrical model. Current is estimated proportionally from absolute simulated joint torque relative to the configured torque limit. Actual servo current depends on the real actuator, voltage, speed, load, controller, and transient behavior.

---

## Foot Contact Switches: Simulation and Hardware

### In PyBullet

The current simulation emulates four binary switches using collision contacts on these link indices:

```python
FOOT_LINK_INDICES = [2, 5, 8, 11]
```

The contact reader checks that the contacting body is one of the known terrain bodies and that the contact normal points sufficiently upward. It does not use contact-force magnitude as a feedback signal.

These indices must match the actual loaded URDF. If the URDF's joint/link ordering changes, update the indices accordingly.

### On the physical robot

The intended hardware interface is four digital contact inputs—one per foot. The controller's conceptual input is simply:

```python
contact[i]  # True if the corresponding foot switch is closed; otherwise False
```

A real implementation would replace the PyBullet contact-reading function with GPIO reads and preserve the per-leg feedback logic.

For hardware integration:

- Confirm whether each switch is active-high or active-low.
- Use suitable electrical pull-ups/pull-downs and input protection for the chosen controller.
- Debounce the mechanical switch in software or hardware.
- Ensure the switch is mechanically actuated by ground contact without being damaged by impact.
- Consider wiring faults and disconnected switches; a binary input alone cannot distinguish every fault from a genuine no-contact state.
- Validate at low speed before using aggressive stride settings.

The simulation's contact detection is an approximation of a physical switch. A real switch's mounting position, travel, hysteresis, and actuation force may produce different contact timing.

---

## Actuator Telemetry

The dashboard reports:

- **Estimated total servo current** in amperes.
- A status band: green below `30 A`, orange from `30 A` to below `45 A`, and red at or above `45 A`.
- **Instantaneous horizontal base speed** calculated from PyBullet's base linear velocity in X and Y.
- **Rolling 5-second average speed** based on timestamped samples.

The telemetry pipeline is split into three parts:

1. The main PyBullet loop reads joint torques and base velocity.
2. A calculation thread estimates current and applies exponential smoothing.
3. A separate Matplotlib process renders the dashboard.

If the plot process falls behind, queued plot samples may be discarded rather than blocking the simulation.

The current display is an approximate model based on torque utilization. It is not a substitute for measuring current on the physical robot.

---

## Important Implementation Notes

- **URDF compatibility:** the robot model must expose the expected 12 actuated joints in the expected order, with the final leg links corresponding to `FOOT_LINK_INDICES`.
- **Terrain detection:** the simulation's contact switch accepts the plane and the ramps created by `_setup_world()`. If you add terrain objects, add their body IDs to `self.terrain_ids`.
- **Simulation timestep:** PyBullet is configured with a `1/240 s` physics timestep, while the controller loop sleeps for approximately `1/60 s`. The current loop calls `stepSimulation()` once per controller iteration. Therefore, each loop iteration advances one configured physics step; it does not automatically execute four physics substeps. Keep this in mind when interpreting simulated elapsed time and dynamics.
- **Contact is binary:** the feedback controller uses contact/no-contact state, not force or pressure magnitude.
- **Joint and link indices are model-specific:** changing the URDF can invalidate the hard-coded indices.
- **Jumping is scripted:** the jump state machine is a sequence of timed body/leg offsets, not a contact-planned dynamic jump controller.
- **Telemetry is estimated:** the displayed current is derived from simulated torque and configured actuator limits.

---

## Troubleshooting

### `FileNotFoundError` or PyBullet cannot load `robot.urdf`

Run the script from the directory containing `robot.urdf`, or update the path passed to `pyb.loadURDF()`. Also check that any meshes referenced by the URDF are available.

### Contact feedback never detects a foot

Check that:

- `FOOT_LINK_INDICES` point to the intended foot/calf collision links.
- The relevant terrain body IDs are in `self.terrain_ids`.
- The URDF has collision geometry on those links.
- The foot collision geometry actually reaches the terrain.
- The contact normal filter is appropriate for the terrain orientation.

### Contact feedback triggers on the wrong links

Inspect the loaded URDF's joint/link order and update `FOOT_LINK_INDICES`. PyBullet's link indices are tied to the loaded model's joint ordering.

### Matplotlib animation warning

A warning such as “Animation was deleted without rendering anything” may appear if the animation object is not retained or the plotting backend/process fails to initialize as expected. The script retains the `FuncAnimation` object in a local variable. If the warning persists, check the Matplotlib GUI backend and whether the plot process is starting successfully.

### The telemetry window does not open

Confirm that Matplotlib is installed and that the selected GUI backend is available. The script selects `TkAgg`, which requires a working Tk installation in the Python environment.

### The robot behaves differently after changing leg dimensions

Update both the controller constants and the corresponding URDF geometry. Check the body-height slider, spawn height, joint limits, collision geometry, and reachable workspace.

---

## Safety Notes

This is a simulation controller and a starting point for hardware development—not a validated safety-critical control system.

Before transferring it to a physical robot:

- Verify joint directions, limits, and zero positions with the legs safely supported.
- Test one actuator and one leg at a time.
- Use a current-limited power supply during initial testing.
- Provide a physical power cutoff appropriate for the actuator system.
- Begin with low command magnitudes and conservative stride settings.
- Confirm that a disconnected or stuck contact switch cannot cause an unsafe motion.
- Validate thermal, electrical, mechanical, and stability limits independently of the simulated telemetry.

**Do not treat simulated speed, current estimates, or contact behavior as guaranteed real-world performance.**

---

## License

No license is specified in the supplied controller source. Add a license file and update this section if you intend to distribute the project.
