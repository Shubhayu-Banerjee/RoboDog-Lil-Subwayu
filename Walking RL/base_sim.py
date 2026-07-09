import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class AcousticSimulation:
    def __init__(self):
        # Physics Constants
        self.v_sound = 343.0  # Speed of sound in m/s
        self.spk1_pos = np.array([-2.0, 0.0])  # Left speaker (x, y)
        self.spk2_pos = np.array([2.0, 0.0])  # Right speaker (x, y)
        self.mic_pos = np.array([0.0, 5.0])  # Draggable microphone (x, y)
        self.spk2_phase = 0.0  # Phase of speaker 2 (0 or pi)

        # Setup Figure and Axes
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('Beats & Interference Simulator')

        # Ax1: Spatial Map (Slightly pushed up to clear room below)
        self.ax_map = self.fig.add_axes([0.1, 0.58, 0.8, 0.38])
        self.ax_map.set_xlim(-10, 10)
        self.ax_map.set_ylim(-2, 12)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title("Spatial Map (Drag the Green Microphone)")
        self.ax_map.set_xlabel("X Position (m)")
        self.ax_map.set_ylabel("Y Position (m)")
        self.ax_map.grid(True, linestyle='--', alpha=0.6)

        # Plot Speakers and Mic
        self.ax_map.plot(*self.spk1_pos, 'ks', markersize=12, label='Speaker 1')
        self.ax_map.plot(*self.spk2_pos, 'bs', markersize=12, label='Speaker 2')
        self.mic_marker, = self.ax_map.plot(*self.mic_pos, 'go', markersize=10, label='Microphone')
        self.ax_map.legend(loc='upper right')

        # Ax2: Waveform Oscilloscope (Centered with adjusted padding)
        self.ax_wave = self.fig.add_axes([0.1, 0.28, 0.8, 0.2])
        self.ax_wave.set_title("Waveform Received at Microphone")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.set_ylim(-2.5, 2.5)
        self.ax_wave.grid(True)

        self.t = np.linspace(0, 0.1, 1000)  # 0.1 seconds of audio
        self.wave_line, = self.ax_wave.plot(self.t, np.zeros_like(self.t), 'r-')

        # --- REPOSITIONED UI CONTROLS TO PREVENT OVERLAP ---
        # Shortened width to 0.55 so text doesn't slam into the button
        self.ax_freq1 = self.fig.add_axes([0.18, 0.13, 0.52, 0.03])
        self.ax_freq2 = self.fig.add_axes([0.18, 0.06, 0.52, 0.03])

        self.s_freq1 = Slider(self.ax_freq1, 'Freq 1 (Hz)', 400.0, 480.0, valinit=440.0)
        self.s_freq2 = Slider(self.ax_freq2, 'Freq 2 (Hz)', 400.0, 480.0, valinit=440.0)

        # Moved the button further right (0.78) and tweaked width to perfectly isolate it
        self.ax_btn = self.fig.add_axes([0.78, 0.06, 0.14, 0.10])
        self.btn_invert = Button(self.ax_btn, 'Turn Spk 2\n(Invert Phase)')
        # ---------------------------------------------------

        # Event Connections
        self.s_freq1.on_changed(self.update)
        self.s_freq2.on_changed(self.update)
        self.btn_invert.on_clicked(self.toggle_phase)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.dragging = False
        self.update(None)

    def calculate_wave(self):
        f1 = self.s_freq1.val
        f2 = self.s_freq2.val

        # Calculate distances from speakers to microphone
        d1 = np.linalg.norm(self.mic_pos - self.spk1_pos)
        d2 = np.linalg.norm(self.mic_pos - self.spk2_pos)

        # Time delays based on speed of sound
        t1 = d1 / self.v_sound
        t2 = d2 / self.v_sound

        # Wave equations at the microphone: y(t) = sin(2*pi*f*(t - d/v) + phase)
        wave1 = np.sin(2 * np.pi * f1 * (self.t - t1))
        wave2 = np.sin(2 * np.pi * f2 * (self.t - t2) + self.spk2_phase)

        return wave1 + wave2

    def update(self, val):
        combined_wave = self.calculate_wave()
        self.wave_line.set_ydata(combined_wave)
        self.fig.canvas.draw_idle()

    def toggle_phase(self, event):
        # Swap between 0 and pi radians
        if self.spk2_phase == 0.0:
            self.spk2_phase = np.pi
            self.ax_map.plot(*self.spk2_pos, 'rs', markersize=12)  # Turn red to show inversion
        else:
            self.spk2_phase = 0.0
            self.ax_map.plot(*self.spk2_pos, 'bs', markersize=12)  # Turn blue for normal

        self.update(None)

    # --- Mouse Dragging Logic ---
    def on_press(self, event):
        if event.inaxes != self.ax_map: return
        # Check if click is near the microphone
        click_pos = np.array([event.xdata, event.ydata])
        if np.linalg.norm(click_pos - self.mic_pos) < 1.0:
            self.dragging = True

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax_map: return
        self.mic_pos = np.array([event.xdata, event.ydata])
        self.mic_marker.set_data([self.mic_pos[0]], [self.mic_pos[1]])
        self.update(None)


if __name__ == '__main__':
    sim = AcousticSimulation()
    plt.show()