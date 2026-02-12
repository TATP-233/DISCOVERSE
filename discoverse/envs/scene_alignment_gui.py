import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np


class SceneAlignmentGUI:
    """GUI for aligning 3DGS and MuJoCo scenes using Matplotlib sliders and textboxes."""
    
    def __init__(self, mj_model, mj_data, init_qpos, init_pos, init_quat, figsize=(8, 12)):
        """
        Initialize the Scene Alignment GUI.
        
        Args:
            mj_model: MuJoCo model object
            mj_data: MuJoCo data object
            init_qpos: Initial joint positions (array)
            init_pos: Initial body position (array of shape 3)
            init_quat: Initial body quaternion (array of shape 4, wxyz format)
            figsize: Figure size for matplotlib
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.init_qpos = init_qpos
        self.init_pos = init_pos
        self.init_quat = init_quat
        
        self.nu = len(init_qpos)
        self.total_sliders = 3 + 4 + self.nu
        
        self.sliders_pos = []
        self.sliders_quat = []
        self.sliders_joint = []
        self.textboxes = []
        
        self.state = {'is_updating_quat': False}
        
        # Setup figure
        self.fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(left=0.25, bottom=0.1, right=0.95, top=0.95)
        
        self._create_sliders()
        self._setup_callbacks()
        self._create_buttons()
        
        plt.show(block=False)
    
    def _create_slider_textbox(self, idx, label, valmin, valmax, valinit):
        """Create a slider with an accompanying textbox."""
        ax_slider = plt.axes([0.20, 0.92 - idx * (0.85 / self.total_sliders), 0.55, 0.02])
        ax_text = plt.axes([0.80, 0.92 - idx * (0.85 / self.total_sliders), 0.15, 0.02])
        
        slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit)
        textbox = TextBox(ax_text, '', initial=f"{valinit:.3f}")
        
        def submit(text):
            try:
                val = float(text)
                val = np.clip(val, valmin, valmax)
                slider.set_val(val)
            except ValueError:
                pass
            textbox.set_val(f"{slider.val:.3f}")

        textbox.on_submit(submit)
        
        def update_text(val):
            textbox.set_val(f"{val:.3f}")
            
        slider.on_changed(update_text)
        self.textboxes.append(textbox)
        return slider
    
    def _create_sliders(self):
        """Create all sliders for position, quaternion, and joints."""
        idx = 0
        
        # Position Sliders
        for i, label in enumerate(['x', 'y', 'z']):
            s = self._create_slider_textbox(
                idx, f'Pos {label}', 
                self.init_pos[i] - 0.1, 
                self.init_pos[i] + 0.1, 
                self.init_pos[i]
            )
            self.sliders_pos.append(s)
            idx += 1
        
        # Quaternion Sliders
        for i, label in enumerate(['w', 'x', 'y', 'z']):
            s = self._create_slider_textbox(idx, f'Quat {label}', -1.0, 1.0, self.init_quat[i])
            self.sliders_quat.append(s)
            idx += 1
        
        # Joint Sliders
        for i in range(self.nu):
            low = self.mj_model.joint(i).range[0]
            high = self.mj_model.joint(i).range[1]
            s = self._create_slider_textbox(
                idx, 
                self.mj_model.joint(i).name if self.mj_model.joint(i).name else f'Joint {i}', 
                low, high, 
                self.init_qpos[i]
            )
            self.sliders_joint.append(s)
            idx += 1
    
    def _update_pos(self, val):
        """Update body position from sliders."""
        p = np.array([s.val for s in self.sliders_pos])
        self.mj_model.body(1).pos[:] = p

    def _update_quat(self, val):
        """Update body quaternion from sliders with normalization."""
        if self.state['is_updating_quat']:
            return
        self.state['is_updating_quat'] = True
        
        q = np.array([s.val for s in self.sliders_quat])
        norm = np.linalg.norm(q)
        if norm > 1e-6:
            q /= norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        
        self.mj_model.body(1).quat[:] = q
        
        for i, s in enumerate(self.sliders_quat):
            if s.val != q[i]:
                s.set_val(q[i])
        
        self.state['is_updating_quat'] = False

    def _update_joint(self, val):
        """Update joint positions from sliders."""
        for i, s in enumerate(self.sliders_joint):
            self.mj_data.qpos[i] = s.val
    
    def _setup_callbacks(self):
        """Setup slider callback functions."""
        for s in self.sliders_pos:
            s.on_changed(self._update_pos)
        for s in self.sliders_quat:
            s.on_changed(self._update_quat)
        for s in self.sliders_joint:
            s.on_changed(self._update_joint)
    
    def _create_buttons(self):
        """Create Reset and Print buttons."""
        ax_reset = plt.axes([0.25, 0.02, 0.3, 0.05])
        btn_reset = Button(ax_reset, 'Reset')
        btn_reset.on_clicked(self._reset)
        
        ax_print = plt.axes([0.6, 0.02, 0.3, 0.05])
        btn_print = Button(ax_print, 'Print')
        btn_print.on_clicked(self._print_info)
    
    def _reset(self, event):
        """Reset all sliders to initial values."""
        for s in self.sliders_pos:
            s.reset()
        for s in self.sliders_quat:
            s.reset()
        for s in self.sliders_joint:
            s.reset()
    
    def _print_info(self, event):
        """Print current state information."""
        print("-" * 30)
        print(f"Pos:   {self.mj_model.body(1).pos}")
        print(f"Quat:  {self.mj_model.body(1).quat}")
        print(f"Joint: {self.mj_data.qpos[:self.nu]}")
    
    def update(self):
        """Update GUI (call this in your main loop)."""
        plt.pause(0.001)
    
    def get_state(self):
        """Get current state as a dictionary."""
        return {
            'pos': self.mj_model.body(1).pos.copy(),
            'quat': self.mj_model.body(1).quat.copy(),
            'qpos': self.mj_data.qpos[:self.nu].copy()
        }
