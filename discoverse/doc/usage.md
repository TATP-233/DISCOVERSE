+ class `BaseConfig`

    Configuration of the simulation environment, including the following contents:

    - `mjcf_file_path`: The simulation scene file, with the suffix `.xml` or `.mjb`.
    - `timestep`: The time step of the physical simulation, in seconds.
    - `decimation`: Downsampling, and the time for each call to the `step` simulation is $decimation \times timestep$.
    - `sync`: Time synchronization. When set to `True`, it will perform sleep during `step` to keep the time speed of the simulation consistent with the real world. It is recommended to set it to `True` during teleoperation and `False` during automatic data generation, which will speed up the data generation.
    - `headless`: Headless mode. If set to `True`, the visualization window will not be displayed. It is recommended to set it to `True` on devices without a display or during automatic data generation.
    - `render_set`: Of dictionary type, used to set the frame rate, width, and height of the rendered image.
    - `obs_rgb_cam_id`: List of integers, used to set the ID of the RGB image acquisition camera.
    - `obs_depth_cam_id`: List of integers, used to set the ID of the depth map acquisition camera.
    - `use_gaussian_renderer`: When set to `True`, 3dgs is used for high-fidelity rendering, otherwise the mujoco native renderer is used.
        The following options are unique to high-fidelity rendering and do not need to be set when using the mujoco native renderer:
    - `rb_link_list`: The body name of the robot.
    - `obj_list`: The body name of the manipulated objects in the scene. Only objects appearing in `rb_link_list` and `obj_list` will appear during 3dgs rendering.
    - `gs_model_dict`: Of dictionary type, where the key is the body name and the value is the path of the corresponding 3dgs ply file.

+ `step`

    The agent interacts with the environment through the `step()` function, executes an action, and receives the next observation, privileged observation, reward, done flag, and other additional information.

    ```python
    observation, privileged_observation, reward, done, info = env.step(action)
    ```

### Tools

There are some commonly used Python scripts in the `scripts` path:

- `convex_decomposition.ipynb`: [Convex decomposition of objects](doc/convex decomposition.md)
- `urdf format`: Format the urdf file.
- `gaussainSplattingConvert.py`: Convert 3dgs ply models between binary and ASCII encoding.
- `gaussainSplattingTranspose.py`: Translate, rotate, and scale a single 3dgs ply model.

Other tools:

- [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf): Convert obj files to mjcf format.
- View the mujoco scene in the terminal
    ```bash
    python3 -m mujoco.viewer --mjcf=<PATH-TO-MJCF-FILE>
    e.g.
    cd models/mjcf
    python3 -m mujoco.viewer --mjcf=mmk2_floor.xml
    ```