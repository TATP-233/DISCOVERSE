"""
A logger class for quadrotor simulation.

Author: Weibin Gu, 4-Nov-2024.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # on macOS
import matplotlib.pyplot as plt
from datetime import datetime
import json

class QuadrotorLogger:
    def __init__(self):
        """ Initialize lists to log the quadrotor states and target states over time """
        self.log = {
            "pos": [], "vel": [], "quat": [], "pqr": [], "eul": [],
            "target_pos": [], "target_eul": [], "target_vel": [], 
            "target_pqr": [], "target_acc": []
        }

    def log_quadrotor_state(self, pos, vel, quat, pqr, eul, target_pos, target_eul, target_vel, target_pqr, target_acc):
        """ Append the current states to the log """
        self.log["pos"].append(pos.copy())
        self.log["vel"].append(vel.copy())
        self.log["quat"].append(quat.copy())
        self.log["pqr"].append(pqr.copy())
        self.log["eul"].append(eul.copy())

        """ Append the target states to the log """
        self.log["target_pos"].append(target_pos.copy())
        self.log["target_eul"].append(target_eul.copy())
        self.log["target_vel"].append(target_vel.copy())
        self.log["target_pqr"].append(target_pqr.copy())
        self.log["target_acc"].append(target_acc.copy())

    def save_logs_to_file(self, filename):
        """ Save logged data to a text file """
        with open(filename, 'w') as f:
            for key, values in self.log.items():
                f.write(f"{key}:\n")
                for v in values:
                    f.write(f"{v}\n")
                f.write("\n")

    def plot_states(self):
        """ States """
        pos = np.array(self.log["pos"])
        vel = np.array(self.log["vel"])
        eul = np.array(self.log["eul"])
        pqr = np.array(self.log["pqr"])
        
        """ Targets """
        target_pos = np.array(self.log["target_pos"])
        target_vel = np.array(self.log["target_vel"])
        target_eul = np.array(self.log["target_eul"])
        target_pqr = np.array(self.log["target_pqr"])

        """ Create subplots """
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        fig.suptitle('Quadrotor States vs. Target States')

        # Position plot
        axes[0].plot(pos[:, 0], label='Pos X')
        axes[0].plot(pos[:, 1], label='Pos Y')
        axes[0].plot(pos[:, 2], label='Pos Z')
        axes[0].plot(target_pos[:, 0], '--', label='Target Pos X')
        axes[0].plot(target_pos[:, 1], '--', label='Target Pos Y')
        axes[0].plot(target_pos[:, 2], '--', label='Target Pos Z')
        axes[0].set_ylabel("Position")
        axes[0].legend()

        # Velocity plot
        axes[1].plot(vel[:, 0], label='Vel X')
        axes[1].plot(vel[:, 1], label='Vel Y')
        axes[1].plot(vel[:, 2], label='Vel Z')
        axes[1].plot(target_vel[:, 0], '--', label='Target Vel X')
        axes[1].plot(target_vel[:, 1], '--', label='Target Vel Y')
        axes[1].plot(target_vel[:, 2], '--', label='Target Vel Z')
        axes[1].set_ylabel("Velocity")
        axes[1].legend()

        # Euler angles plot
        axes[2].plot(np.rad2deg(eul[:, 0]), label='Roll')
        axes[2].plot(np.rad2deg(eul[:, 1]), label='Pitch')
        axes[2].plot(np.rad2deg(eul[:, 2]), label='Yaw')
        axes[2].plot(np.rad2deg(target_eul[:, 0]), '--', label='Target Roll')
        axes[2].plot(np.rad2deg(target_eul[:, 1]), '--', label='Target Pitch')
        axes[2].plot(np.rad2deg(target_eul[:, 2]), '--', label='Target Yaw')
        axes[2].set_ylabel("Euler Angles [deg]")
        axes[2].legend()

        # Angular rate (pqr) plot
        axes[3].plot(np.rad2deg(pqr[:, 0]), label='p')
        axes[3].plot(np.rad2deg(pqr[:, 1]), label='q')
        axes[3].plot(np.rad2deg(pqr[:, 2]), label='r')
        axes[3].plot(np.rad2deg(target_pqr[:, 0]), '--', label='Target p')
        axes[3].plot(np.rad2deg(target_pqr[:, 1]), '--', label='Target q')
        axes[3].plot(np.rad2deg(target_pqr[:, 2]), '--', label='Target r')
        axes[3].set_ylabel("Angular Rates [deg/sec]")
        axes[3].legend()

        plt.xlabel("Time Steps")
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        # plt.show()

        """ Save plots """
        now = datetime.now()
        timestamp = now.strftime("discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S")
        filename = f"{timestamp}_states.png"
        plt.savefig(filename)

    def plot_3d_position(self):
        """ Plot 3D position with target points """
        pos = np.array(self.log["pos"])
        target_pos = np.array(self.log["target_pos"])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="Actual Path", color='b')
        ax.scatter(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], color='r', marker='x', label="Target Points", s=50)
        
        ax.set_title("3D Position Tracking")
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_zlabel("Z Position [m]")
        ax.legend()

        # Save plot
        now = datetime.now()
        timestamp = now.strftime("discoverse/examples/skyrover_on_rm2car/log/%Y-%m-%d %H:%M:%S")
        filename = f"{timestamp}_3d_position.png"
        plt.savefig(filename)
