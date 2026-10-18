# Bundle role — Robotics data & fleet loop (deferred)

Portfolio-level decision: **deferred**. See
[`Mullassery/BUNDLES.md`](https://github.com/Mullassery/Mullassery/blob/main/BUNDLES.md).

- **Role:** Robot-learning dataset loading/conversion
- **Upstream/downstream:** PyRoboReplay, PyRoboSimulator, training pipelines
- **Unlock condition:** Linux/CUDA wheels, then a shared LeRobot/ROS 2 demo dataset.
- **Current status:** standalone; not a dependency of the other bundle members.

Do not add cross-repo runtime dependencies until the unlock condition is met.
When it is, add a versioned interface (not a git URL), a shared
observation/mission schema, and an end-to-end demo.
