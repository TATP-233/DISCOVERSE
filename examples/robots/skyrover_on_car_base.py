"""
A class for cooperative aerial-ground robot team.

Robot team (1 SkyRover, 1 AIRBOT): `cooperative_aerial_ground_sim.xml`.

Author: Weibin Gu, 8-Nov-2024.
"""
from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
import mujoco
import numpy as np
import time
import xml.etree.ElementTree as ET

DEBUGGING_LINE = False

class SkyRoverOnCarBase(SimulatorBase):
    def __init__(self, config: BaseConfig):
        super().__init__(config)

        """ Multi-robot interface"""
        self.robot_names = ["skyrover", "rm2"] # robot entities
        self.robots = {name: {"joints": {}, "bodies": {}, "sites": {}} for name in self.robot_names} # robot attributes
        self.actuator_mapping = self._parse_actuators_from_xml(self.mjcf_file) # robot attributes

        # Initialize joint and body mappings for each robot
        for robot_name in self.robot_names:
            self._initialize_robot(robot_name)
        print("="*100)
        print("Robots initialized with the following mappings:", self.robots)
        print("="*100)

        # SkyRover
        self.locomode = "drone" # locomotion mode
        self.switching_time = None # timer for locomotion switching
        self.switch_locomotion(locomode=self.locomode) # set to drone mode by default

        # RM2
        self.rm2_wheel_radius = 0.1
        self.rm2_robot_radius=0.184*2

        self.dt = self.mj_model.opt.timestep # sampling time [sec]

    def updateControl(self, action):
        self.mj_data.ctrl[:] = action[:]

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"  : self.mj_data.qpos.tolist(),
            "jv"  : self.mj_data.qvel.tolist(),
            "img" : self.img_rgb_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None
    
    ###########################################################################
    ### Setter function #######################################################
    ########################################################################### 

    def set_joint_position(self, robot_name, joint_name, position):
        """
        Set the position of a specific joint of a robot.
        
        TODO: Now all the joint indices are manually set (can be easily get from Mujoco > Control Panel). This means that when changing XML file, this function has to adapt accordingly.

        Args:
            robot_name (str): The name of the robot.
            joint_name (str): The name of the joint.
            position (float): The desired position value for the joint.
        """
        if self._sanity_check(robot_name=robot_name,
                              attribute_name=joint_name,
                              type="joints"):
            if robot_name == "skyrover":
                if joint_name == "skyrover_stretch_joint":
                    self.mj_data.ctrl[0] = position
                if joint_name == "skyrover_folder1_joint":
                    self.mj_data.ctrl[1] = position
                if joint_name == "skyrover_folder2_joint":
                    self.mj_data.ctrl[6] = position
                if joint_name == "skyrover_rotor1_joint":
                    self.mj_data.ctrl[2] = position
                if joint_name == "skyrover_rotor2_joint":
                    self.mj_data.ctrl[3] = position
                if joint_name == "skyrover_rotor3_joint":
                    self.mj_data.ctrl[7] = position
                if joint_name == "skyrover_rotor4_joint":
                    self.mj_data.ctrl[8] = position 
            if robot_name == "rm2":
                if joint_name == "rm2_wheel_joint1":
                    self.mj_data.ctrl[15] = position
                if joint_name == "rm2_wheel_joint2":
                    self.mj_data.ctrl[16] = position
                if joint_name == "rm2_wheel_joint3":
                    self.mj_data.ctrl[17] = position
                if joint_name == "rm2_wheel_joint4":
                    self.mj_data.ctrl[18] = position 
        else:
            print(f"[ERROR] Sanity check failed for robot '{robot_name}' and attribute '{joint_name}'. Check that the attribute exists and the robot is initialized properly.")

    def set_site_control_value(self, robot_name, site_name, control_value):
        """
        Set the control value for an actuator associated with a specific site and prefixed by the robot's name.
        
        This function is intended for actuators that control based on site attributes (e.g., rotor thrusts in the SkyRover robot) rather than standard joint attributes. For actuators using joint attributes, obtaining actuator IDs is more straightforward via `mj_id2name`. For additional implementation details, refer to `_parse_actuators_from_xml`.

        Args:
            robot_name (str): The name of the robot.
            site_name (str): The name of the site.
            control_value (float): The desired value for the site.
        """
        # Check if the site_name starts with the specified robot_name prefix
        if not site_name.startswith(robot_name):
            print(f"[ERROR] Site '{site_name}' does not start with the robot name prefix '{robot_name}'.")
            return

        # Retrieve the actuator ID associated with the site
        actuator_id = self._get_actuator_id_from_site(site_name)
        
        # Check if actuator ID was found and set the control value if so
        if actuator_id is not None:
            self.mj_data.ctrl[actuator_id] = control_value
            # print(f"Set {robot_name} control value {control_value} to actuator at site '{site_name}' with actuator ID '{actuator_id}'") # Debuggine line
        else:
            print(f"[ERROR] Actuator for site '{site_name}' not found.")


    ###########################################################################
    ### Getter function #######################################################
    ###########################################################################

    ### For Joints ############################################################
    def get_joint_position(self, robot_name, joint_name):
        if self._sanity_check(robot_name=robot_name,
                              attribute_name=joint_name,
                              type="joints"):
            joint_id = self.robots[robot_name]["joints"][joint_name]
            joint_index = self.mj_model.jnt_qposadr[joint_id]
            return self.mj_data.qpos[joint_index]
        else:
            print(f"[ERROR] Sanity check failed for robot '{robot_name}' and attribute '{joint_name}'. Check that the attribute exists and the robot is initialized properly.")
            return

    def get_joint_velocity(self, robot_name, joint_name):
        if self._sanity_check(robot_name=robot_name,
                              attribute_name=joint_name,
                              type="joints"):
            joint_id = self.robots[robot_name]["joints"][joint_name]
            joint_index = self.mj_model.jnt_dofadr[joint_id]
            return self.mj_data.qvel[joint_index]
        else:
            print(f"[ERROR] Sanity check failed for robot '{robot_name}' and attribute '{joint_name}'. Check that the attribute exists and the robot is initialized properly.")
            return
    
    ### For Bodies ############################################################
    def get_root_body_pose(self, robot_name, root_body_name):
        """
        Get the position and attitude of the root body of a robot. 
        Args:
            robot_name (str): The name of the robot.
            body_name (str): The name of the body.
        Returns:
            pos (3), quat (4) for root bodies, or scalar depending on joint type.
        """
        if robot_name != root_body_name:
            print(f"[ALERT] The root body name is incorrectly specified. Expected: '{robot_name}', but got: '{root_body_name}'. Please verify the input or robot XML file. Make sure that root body has the same name as the robot name.")

        # Retrieve body ID from the robot dictionary
        body_id = self._get_body_id(robot_name, root_body_name)

        # Retrieve the joint ID that controls this body (if available)
        # for root_body_name, body_id in self.robots[robot_name]["bodies"].items():
            # Find the joint associated with this body
        joint_id = self._get_joint_id_from_body_id(body_id=body_id,
                                                    qpos_qvel_flag="qpos") 
                    
        # If a joint was found, retrieve its position in qpos
        if joint_id is not None:
            pos = self.mj_data.qpos[joint_id:joint_id+3]
            quat = self.mj_data.qpos[joint_id+3:joint_id+7]
            if DEBUGGING_LINE: print(f"Body: {root_body_name} | ID: {body_id} | Position (qpos): {pos}") # Debugging line
            return pos, quat

        print(f"[ERROR] No specific joint found for body '{root_body_name}' or joint type not handled.")
        return None

    def get_root_body_velocity(self, robot_name, root_body_name):
        """
        Get the linear velocity of the root body of a robot.
        Args:
            robot_name (str): The name of the robot.
            body_name (str): The name of the body.
        Returns:
            vel (3), pqr (3) for root bodies.
        """
        if robot_name != root_body_name:
            print(f"[ALERT] The root body name is incorrectly specified. Expected: '{robot_name}', but got: '{root_body_name}'. Please verify the input or robot XML file. Make sure that root body has the same name as the robot name.")

        # Retrieve body ID from the robot dictionary
        body_id = self._get_body_id(robot_name, root_body_name)

        # Retrieve the joint ID that controls this body (if available)
        for root_body_name, body_id in self.robots[robot_name]["bodies"].items():
            # Find the joint associated with this body
            joint_id = self._get_joint_id_from_body_id(body_id=body_id,
                                                        qpos_qvel_flag="qvel")  
            # If a joint was found, retrieve its velocity in qvel
            if joint_id is not None:
                vel = self.mj_data.qvel[joint_id:joint_id+3]
                pqr = self.mj_data.qvel[joint_id+3:joint_id+6]
                if DEBUGGING_LINE: print(f"Body: {root_body_name} | ID: {body_id} | Velocity (qvel): {vel}") # Debugging line
                return vel, pqr
        
        print(f"[ERROR] No specific free joint found for body '{root_body_name}' to get linear velocity.")
        return None

    ###########################################################################
    ### Private method ########################################################
    ###########################################################################

    def _parse_actuators_from_xml(self, xml_file):
        """
        Parse the XML to manually map actuator indices to sites.
        This is a workaround that circumvents the need for `mj_id2name` for actuators with sites. 
        
        MuJoCo doesn't find the names because actuators that use site attributes (rather than joint attributes) are not directly linked to body IDs in a way that `mj_id2name` can access. MuJoCo expects actuators to be associated with a joint attribute to create a clear mapping for `mj_id2name` calls, but the site parameter does not directly map to an ID within MuJoCo's internal structure.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        actuator_mapping = {}

        # Iterate over motor actuators in XML
        for i, motor in enumerate(root.findall(".//actuator/motor")):
            site_name = motor.get("site")  # get the site linked to this actuator
            
            # Check if the site name is valid
            if site_name:
                actuator_mapping[site_name] = i  # Map site name to index

        # print("Manual mapping between actuators and sites: ", actuator_mapping) # Debugging line
        return actuator_mapping

    def _initialize_robot(self, robot_name):
        """
        Set up joint, body, site mappings for each robot.
        """
        # Initialize joints
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            body_id = self.mj_model.jnt_bodyid[i]  # Get the body ID for this joint
            # print(f"Joint ID {i}: {joint_name}")  # Debugging line: print all joint names
            if joint_name and joint_name.startswith(robot_name):
                self.robots[robot_name]["joints"][joint_name] = i
        # print(f"Initialized joints for {robot_name}: {self.robots[robot_name]['joints']}")  # Debugging line

        # Initialize bodies
        for i in range(self.mj_model.nbody):
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and body_name.startswith(robot_name):
                self.robots[robot_name]["bodies"][body_name] = i
        # print(f"Initialized bodies for {robot_name}: {self.robots[robot_name]['bodies']}")  # Debugging line

        # Initialize sites (for control, we need actuator ID not site ID)
        for i in range(self.mj_model.nsite):
            site_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
            if site_name and site_name.startswith(robot_name):
                self.robots[robot_name]["sites"][site_name] = i
        # print(f"Initialized sites for {robot_name}: {self.robots[robot_name]['sites']}")  # Debugging line

        # Error handling if no bodies or joints are found
        if not self.robots[robot_name]["bodies"] or not self.robots[robot_name]["joints"]:
            print("[ERROR] To properly manage joints and other properties for multiple robots using this script,",
            "ensure that the bodies and joints elements in the XML file are prefixed with the robot name.")

    def _sanity_check(self, robot_name, attribute_name, type):
        """ 
        Sanity check for joints and sites in robot's body. 
        """
        # Check if the robot exists
        if robot_name not in self.robots:
            print(f"[ERROR] Robot {robot_name} not found.")
            return False

        # Check if the attribute exists in the robot
        if type not in ["joints", "sites"]:
            raise TypeError("[ERROR] Such attribute type does not exist!")
        else:
            if attribute_name not in self.robots[robot_name][type]:
                print(f"[ERROR] Site {attribute_name} not found for robot {robot_name}.")
                return False
            else:
                return True
            
    def _get_actuator_id_from_site(self, site_name):
        """
        Retrieve the actuator ID based on the site name.
        """
        actuator_id = self.actuator_mapping.get(site_name)
        if actuator_id is not None:
            return actuator_id
        else:
            print(f"[ERROR] Actuator for site '{site_name}' not found.")
            return None
    
    def _get_body_id(self, robot_name, body_name):
        """
        Get the body_id for a specific body from the robot's body mapping.
        """
        body_id = self.robots[robot_name]["bodies"][body_name]
        if body_id is None:
            print(f"[ERROR] Body '{body_name}' not found for robot '{robot_name}'.")
            return None
        else:
            return body_id
        
    def _get_joint_id_from_body_id(self, body_id, qpos_qvel_flag):
        """
        Find the joint associated with this body based on the given flag.
        The flag indicates whether to retrieve the joint's position or velocity data.

        Specifically, we use the attribute `jnt_qposadr` which maps each joint ID to its starting index in qpos. This mapping is used to determine which qpos indices control a specific joint. By extension, it can also be determined how each body connected to that joint will move.
        
        Args:
            body_id: The ID of the body to look up.
            qpos_qvel_flag: Either "qpos" or "qvel", indicating whether to retrieve position or velocity.

        Returns:
            joint_id: The start address of the joint's data in either `qpos` or `qvel`.
        """

        # Mapping for qpos and qvel
        joint_addr_map = {
            "qpos": self.mj_model.jnt_qposadr,  # Joint position data address
            "qvel": self.mj_model.jnt_dofadr    # Joint velocity data address
        }

        # Validate the flag
        if qpos_qvel_flag not in joint_addr_map:
            raise ValueError(f"[ERROR] Invalid flag: `{qpos_qvel_flag}`.")
        
        # Get the corresponding joint address map for qpos or qvel
        joint_addr = joint_addr_map[qpos_qvel_flag]
        
        # Iterate over the joints to find the associated body
        for j_id, b_id in enumerate(self.mj_model.jnt_bodyid):  # jnt_bodyid: ID of the body's joint
            if b_id == body_id:
                return joint_addr[j_id]  # Return the start index for the joint in `qpos` or `qvel` data structure
        
        # If no joint is associated with the body
        print("Not available (no associated joint)")
        return None
    
    ###########################################################################
    ### High-level method #####################################################
    ###########################################################################
    def switch_locomotion(self, locomode="drone", switching_duration=5):
        """
        Switch the locomotion of the SkyRover robot. The default mode is "drone" mode.
        
        Args:
            locomode: locomotion mode, "drone" or "rover"
        """

        """ Joint limits """
        stretch_joint_drone = -5.04e-2
        folder1_joint_drone = 2.5e-4
        folder2_joint_drone = 2.5e-4
        stretch_joint_rover = -9.1e-2
        folder1_joint_rover = 1.50
        folder2_joint_rover = 1.50

        ready_flag = False
        if locomode == "drone":
            """ Switch to drone mode """

            # Compute joint position increments
            folder_joint_increment, self.switching_time = self._incremental_change(79.8, 0, switching_duration, "decrease", self.switching_time)
            stretch_joint_increment, self.switching_time = self._incremental_change(-545, 0, switching_duration, "increase", self.switching_time)

            if self.get_joint_position("skyrover", "skyrover_stretch_joint") < stretch_joint_drone:
                self.set_joint_position("skyrover", "skyrover_stretch_joint", stretch_joint_increment)
            if self.get_joint_position("skyrover", "skyrover_folder1_joint") > folder1_joint_drone:
                self.set_joint_position("skyrover", "skyrover_folder1_joint", folder_joint_increment)
            if self.get_joint_position("skyrover", "skyrover_folder2_joint") > folder2_joint_drone:
                self.set_joint_position("skyrover", "skyrover_folder2_joint", folder_joint_increment)
            if self.get_joint_position("skyrover", "skyrover_stretch_joint") >= stretch_joint_drone \
            and self.get_joint_position("skyrover", "skyrover_folder1_joint") <= folder1_joint_drone \
            and self.get_joint_position("skyrover", "skyrover_folder2_joint") <= folder2_joint_drone:
                ready_flag = True
                self.locomode = "drone"
                self.switching_time = None # reset
        elif locomode == "rover":
            """ Switch to rover mode """
            
            # Compute joint position increments
            folder_joint_increment, self.switching_time = self._incremental_change(0, 79.8, switching_duration, "increase", self.switching_time)
            stretch_joint_increment, self.switching_time = self._incremental_change(0, -545, switching_duration, "decrease", self.switching_time)

            if self.get_joint_position("skyrover", "skyrover_folder1_joint") < folder1_joint_rover:
                self.set_joint_position("skyrover", "skyrover_folder1_joint", folder_joint_increment)
            if self.get_joint_position("skyrover", "skyrover_folder2_joint") < folder2_joint_rover:
                self.set_joint_position("skyrover", "skyrover_folder2_joint", folder_joint_increment)
                
            # To avoid undesirable motion of wheels due to the friction between the floor,
            # rotate two folder links first, followed by the sliding motion of the stretch link.
            if self.get_joint_position("skyrover", "skyrover_folder1_joint") >= folder1_joint_rover \
            and self.get_joint_position("skyrover", "skyrover_folder2_joint") >= folder2_joint_rover:
                if self.get_joint_position("skyrover", "skyrover_stretch_joint") > stretch_joint_rover:
                    self.set_joint_position("skyrover", "skyrover_stretch_joint", stretch_joint_increment)
            if self.get_joint_position("skyrover", "skyrover_stretch_joint") <= stretch_joint_rover \
            and self.get_joint_position("skyrover", "skyrover_folder1_joint") >= folder1_joint_rover \
            and self.get_joint_position("skyrover", "skyrover_folder2_joint") >= folder2_joint_rover:
                self.set_joint_position("skyrover", "skyrover_wheel1_joint", 0)
                self.set_joint_position("skyrover", "skyrover_wheel2_joint", 0)
                self.set_joint_position("skyrover", "skyrover_wheel3_joint", 0)
                self.set_joint_position("skyrover", "skyrover_wheel4_joint", 0)
                ready_flag = True
                self.locomode = "rover"
                self.switching_time = None # reset
        else:
            raise TypeError("[ERROR] Non-existing locomotion mode. ['drone', 'rover'] are the only available options for now.")
        
        return ready_flag
    
    def get_skyrover_state(self, verbose=False, robot="skyrover"):
        """ 
        Get SkyRover robot's state (groud truth in the simulator). 

        Args:
            verbose: set "True" to print robot's state information.
        """
        pos, quat = self.get_root_body_pose("skyrover", robot) # robot position (x, y, z), robot orientation (w, x, y, z)
        vel, pqr = self.get_root_body_velocity("skyrover", robot) # robot linear velocity (x, y, z), body-fixed angular rate (p, q, r)

        # Convert quaternion to Euler angles
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(quat[[1, 2, 3, 0]])  # convert to (x, y, z, w) format
        eul = rotation.as_euler('ZYX', degrees=False)[::-1]  # roll, pitch, yaw in radians

        # Print state information
        if verbose:
            x, y, z = pos
            vx, vy, vz = vel
            roll, pitch, yaw = np.rad2deg(eul)  # euler angle in degrees
            p, q, r = np.rad2deg(pqr) # body rate in degrees
            print(f"SkyRover Robot's State:")
            print(f"  Position (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  Velocity (vx, vy, vz): ({vx:.2f}, {vy:.2f}, {vz:.2f})")
            print(f"  Euler Angle (roll, pitch, yaw): {roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°")
            print(f"  Body Rate (p, q, r): ({p:.2f}°/s, {q:.2f}°/s, {r:.2f}°/s)")

        return pos, vel, quat, pqr, eul
    
    def skyrover_drive(self, diff_drive_ctrl):
        """
        Motion control of the SkyRover robot in rover mode.

        Args:
            diff_drive_ctrl: a list of length 4, [front left, back left, back right, front right]
        """
        if self.locomode != "rover":
            raise ValueError("[ERROR] Driving command can be executed only when SkyRover is in rover mode.")
        else:
            self.set_joint_position("skyrover", "skyrover_rotor1_joint", diff_drive_ctrl[0])
            self.set_joint_position("skyrover", "skyrover_rotor2_joint", diff_drive_ctrl[1])
            self.set_joint_position("skyrover", "skyrover_rotor3_joint", diff_drive_ctrl[2])
            self.set_joint_position("skyrover", "skyrover_rotor4_joint", diff_drive_ctrl[3])

    def skyrover_fly(self, rotor_thrust, robot):
        """
        Motion control of the SkyRover robot in drone mode.

        Args:
            rotor_thrust: a list of length 4, [front left (CCW), back left (CW), back right (CCW), front right (CW)]
        """
        if self.locomode != "drone":
            raise ValueError("[ERROR] Flying command can be executed only when SkyRover is in drone mode.")
        else:
            thrust_fl = f"{robot}_thrust_fl"
            thrust_bl = f"{robot}_thrust_bl"
            thrust_br = f"{robot}_thrust_br"
            thrust_fr = f"{robot}_thrust_fr"
            self.set_site_control_value("skyrover", thrust_fl, rotor_thrust[0])
            self.set_site_control_value("skyrover", thrust_bl, rotor_thrust[1])
            self.set_site_control_value("skyrover", thrust_br, rotor_thrust[2])
            self.set_site_control_value("skyrover", thrust_fr, rotor_thrust[3])
            # print(f"skyrover_fly: {rotor_thrust[0], rotor_thrust[1], rotor_thrust[2], rotor_thrust[3]}")

    def get_rm2_state(self, verbose=False):
        """ 
        Get robot's state (groud truth in the simulator). 

        Args:
            verbose: set "True" to print robot's state information.
        """
        pos, quat = self.get_root_body_pose("rm2", "rm2") # robot position (x, y, z), robot orientation (w, x, y, z)
        vel, pqr = self.get_root_body_velocity("rm2", "rm2") # robot linear velocity (x, y, z), body-fixed angular rate (p, q, r)

        # Convert quaternion to Euler angles
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(quat[[1, 2, 3, 0]])  # convert to (x, y, z, w) format
        eul = rotation.as_euler('ZYX', degrees=False)[::-1]  # roll, pitch, yaw in radians

        # Print state information
        if verbose:
            x, y, z = pos
            vx, vy, vz = vel
            roll, pitch, yaw = np.rad2deg(eul)  # euler angle in degrees
            p, q, r = np.rad2deg(pqr) # body rate in degrees
            print(f"RM2 Robot's State:")
            print(f"  Position (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  Velocity (vx, vy, vz): ({vx:.2f}, {vy:.2f}, {vz:.2f})")
            print(f"  Euler Angle (roll, pitch, yaw): {roll:.2f}°, {pitch:.2f}°, {yaw:.2f}°")
            print(f"  Body Rate (p, q, r): ({p:.2f}°/s, {q:.2f}°/s, {r:.2f}°/s)")

        return pos, vel, quat, pqr, eul

    def rm2_drive(self, diff_drive_ctrl):
        """
        Motion control of the AIRBOT robot.

        Args:
            diff_drive_ctrl: a list of length 4, [front left, front right, back right, back left]
        """
        self.set_joint_position("rm2", "rm2_wheel_joint1", diff_drive_ctrl[0])
        self.set_joint_position("rm2", "rm2_wheel_joint2", diff_drive_ctrl[1])
        self.set_joint_position("rm2", "rm2_wheel_joint3", diff_drive_ctrl[2])
        self.set_joint_position("rm2", "rm2_wheel_joint4", diff_drive_ctrl[3])

    def _incremental_change(self, start_value, target_value, duration=5.0, change_type="increase", start_time=None):
        """
        Incrementally changes a value over time either to a maximum (increase) or minimum (decrease) value.

        Args:
            start_value: The starting value.
            target_value: The target value (either maximum or minimum depending on change_type).
            duration: The total duration (in seconds) for reaching the target value.
            change_type: The type of change, either "increase" or "decrease".
            start_time: The initial start time; if None, it will be initialized within the function.

        Returns:
            current_value: The value after the duration has passed (should be target_value).
        """
        # Ensure the correct direction for change
        if change_type == "increase" and start_value >= target_value:
            raise ValueError("start_value must be less than target_value for an increase.")
        if change_type == "decrease" and start_value <= target_value:
            raise ValueError("start_value must be greater than target_value for a decrease.")

        # Initialize start_time if it's None (first time function call)
        if start_time is None:
            start_time = time.time()

        # Initialize current value
        current_value = start_value

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Calculate how much to change based on elapsed time
        if change_type == "increase":
            change = (elapsed_time / duration) * (target_value - start_value)
            current_value = min(start_value + change, target_value)  # Ensure it does not exceed target_value
        elif change_type == "decrease":
            change = (elapsed_time / duration) * (start_value - target_value)
            current_value = max(start_value - change, target_value)  # Ensure it does not go below target_value

        # Return updated value and start_time for next iteration
        return current_value, start_time