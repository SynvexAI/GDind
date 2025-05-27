import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
import mss
import time
import random
from collections import deque, namedtuple
from pynput.mouse import Button, Controller as MouseController
from pynput import keyboard as pynput_keyboard
import logging
import os
import datetime
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
try:
    import pygetwindow
except ImportError:
    pygetwindow = None
try:
    from pywinauto.application import Application
except ImportError:
    Application = None

import GDint_config as config

gui_tk_root = None
gui_data_lock = threading.Lock()
gui_shared_data = {
    "ai_view": None, "raw_capture_view": None,
    "q_values": [0.0, 0.0], "action": 0, "epsilon": 0.0,
    "episode": 0, "step": 0, "current_episode_reward": 0.0,
    "status_text": "Initializing...", "game_region_info": "N/A",
    "avg_reward_history": deque(maxlen=100), "avg_loss_history": deque(maxlen=100),
    "current_loss": 0.0, "total_steps": 0, "fps_info": "AI: 0 | GUI: 0",
    "is_paused": False
}
stop_event = threading.Event()
pause_event = threading.Event() 
game_region_display_window = None

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=log_format, handlers=[
        logging.FileHandler(config.LOG_FILE, mode='w'),
        logging.StreamHandler()
    ])
    for logger_name in ['matplotlib', 'PIL', 'pygetwindow', 'pynput']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.info(f"Logging for {config.PROJECT_NAME} setup. Level: {config.LOG_LEVEL.upper()}, Device: {config.DEVICE}")

setup_logging()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, h, w, outputs, num_frames_stacked=1, is_grayscale=True):
        super(DQN, self).__init__()
        num_input_channels = (1 if is_grayscale else 3) * num_frames_stacked
        
        self.conv1 = nn.Conv2d(num_input_channels, 32, kernel_size=8, stride=4, padding=2) 
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.bn3 = nn.BatchNorm2d(64)

        def conv_out_size(size, kernel, stride, padding):
            return (size - kernel + 2 * padding) // stride + 1
        
        final_h = conv_out_size(conv_out_size(conv_out_size(h, 8, 4, 2), 4, 2, 1), 3, 1, 1)
        final_w = conv_out_size(conv_out_size(conv_out_size(w, 8, 4, 2), 4, 2, 1), 3, 1, 1)
        linear_input_size = final_h * final_w * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        logging.info(f"DQN: Input ({num_input_channels}, H:{h}, W:{w}), ConvOut ({final_h}, {final_w}), LinearIn: {linear_input_size}, Outputs: {outputs}")

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.head(x)

class GameEnvironment:
    def __init__(self):
        self.sct = mss.mss()
        self.mouse = MouseController()
        self.monitor_region = self._get_and_focus_game_window()

        if self.monitor_region is None:
            logging.error(f"Window '{config.WINDOW_TITLE_SUBSTRING}' not found or unmanageable. Using fallback or first monitor.")
            self.monitor_region = config.FALLBACK_GAME_REGION if config.FALLBACK_GAME_REGION else self.sct.monitors[1]
        
        global gui_shared_data
        with gui_data_lock:
            gui_shared_data["game_region_info"] = f"Region: L{self.monitor_region['left']}, T{self.monitor_region['top']}, W{self.monitor_region['width']}, H{self.monitor_region['height']}"
        logging.info(f"GameEnvironment: Screen region set to {self.monitor_region}")
        
        self.game_over_template = self._load_template(config.GAME_OVER_TEMPLATE_PATH, "Game Over")
        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)
        
        if config.SHOW_GAME_REGION_OUTLINE:
            self._create_region_display_window()

    def _create_region_display_window(self):
        global game_region_display_window
        if game_region_display_window: game_region_display_window.destroy()

        try:
            root = tk.Toplevel()
            root.overrideredirect(True) 
            root.attributes("-topmost", True)
            root.attributes("-alpha", 0.3) 
            root.geometry(f"{self.monitor_region['width']}x{self.monitor_region['height']}+{self.monitor_region['left']}+{self.monitor_region['top']}")
            
            canvas = tk.Canvas(root, width=self.monitor_region['width'], height=self.monitor_region['height'], bg='gray', highlightthickness=0) 
            canvas.pack()
            canvas.create_rectangle(
                config.GAME_REGION_OUTLINE_THICKNESS // 2,
                config.GAME_REGION_OUTLINE_THICKNESS // 2,
                self.monitor_region['width'] - config.GAME_REGION_OUTLINE_THICKNESS // 2,
                self.monitor_region['height'] - config.GAME_REGION_OUTLINE_THICKNESS // 2,
                outline=config.GAME_REGION_OUTLINE_BORDER_COLOR,
                width=config.GAME_REGION_OUTLINE_THICKNESS
            )
            
            if os.name == 'nt':
                try:
                    import win32gui
                    import win32con
                    hwnd = root.winfo_id()
                    styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                    styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
                    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
                    
                    
                except ImportError:
                    logging.warning("pywin32 not installed, cannot make region outline click-through on Windows.")
                except Exception as e:
                    logging.warning(f"Error making region outline click-through: {e}")

            game_region_display_window = root
            logging.info("Game region outline display created.")
        except Exception as e:
            logging.error(f"Failed to create game region display window: {e}")


    def _load_template(self, path, name):
        if not path or not os.path.exists(path):
            logging.warning(f"{name} template not specified or not found: {path}")
            return None
        template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if template is None:
            logging.error(f"Failed to load {name} template from {path}")
            return None
        
        if config.GRAYSCALE:
            if len(template.shape) == 3 and template.shape[2] == 4: 
                template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            elif len(template.shape) == 3: 
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        elif len(template.shape) == 3 and template.shape[2] == 4: 
             template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        
        logging.info(f"Loaded {name} template: {path}, Shape: {template.shape}, Grayscale config: {config.GRAYSCALE}")
        return template

    def _get_and_focus_game_window(self):
        if not pygetwindow: return None
        try:
            gd_windows = pygetwindow.getWindowsWithTitle(config.WINDOW_TITLE_SUBSTRING)
            if not gd_windows: return None
            
            gd_window = gd_windows[0]
            if gd_window.isMinimized: gd_window.restore()
            
            if Application and os.name == 'nt':
                try:
                    app = Application().connect(handle=gd_window._hWnd, timeout=5)
                    app_window = app.window(handle=gd_window._hWnd)
                    if app_window.exists() and app_window.is_visible():
                        app_window.set_focus()
                        logging.info(f"Focused window '{gd_window.title}' via pywinauto.")
                    else: gd_window.activate() 
                except Exception: gd_window.activate()
            else: gd_window.activate()
            time.sleep(0.2) 

            return {"top": gd_window.top, "left": gd_window.left, 
                    "width": gd_window.width, "height": gd_window.height,
                    "monitor": 1} 
        except Exception as e:
            logging.error(f"Error in _get_and_focus_game_window: {e}")
            return None

    def _capture_frame_raw_bgr(self):
        try:
            sct_img = self.sct.grab(self.monitor_region)
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        except mss.exception.ScreenShotError as e:
            logging.error(f"Screen capture error: {e}. Retrying...")
            time.sleep(0.05)
            return self._capture_frame_raw_bgr()

    def _preprocess_frame_for_ai(self, frame_bgr):
        if config.GRAYSCALE:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            frame = frame_bgr
        
        frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        return frame_resized.astype(np.uint8)

    def _stack_frames_for_ai(self, processed_frame_for_ai):
        if config.GRAYSCALE:
            frame_chw = np.expand_dims(processed_frame_for_ai, axis=0)
        else:
            frame_chw = np.transpose(processed_frame_for_ai, (2, 0, 1))

        if not self.stacked_frames:
            for _ in range(config.NUM_FRAMES_STACKED):
                self.stacked_frames.append(frame_chw)
        else:
            self.stacked_frames.append(frame_chw)
        
        stacked_state_tensor_data = np.concatenate(list(self.stacked_frames), axis=0)
        return torch.from_numpy(stacked_state_tensor_data).unsqueeze(0).to(config.DEVICE).float()

    def reset(self):
        self.stacked_frames.clear()
        raw_frame_bgr = self._capture_frame_raw_bgr()
        processed_frame = self._preprocess_frame_for_ai(raw_frame_bgr)
        
        if config.GRAYSCALE: frame_chw = np.expand_dims(processed_frame, axis=0)
        else: frame_chw = np.transpose(processed_frame, (2,0,1))
        for _ in range(config.NUM_FRAMES_STACKED): self.stacked_frames.append(frame_chw)
            
        initial_stacked_state = torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float()
        return initial_stacked_state, raw_frame_bgr

    def step(self, action_value):
        if action_value == 1: 
            self.mouse.press(Button.left)
            time.sleep(config.JUMP_DURATION)
            self.mouse.release(Button.left)
        
        if config.ACTION_DELAY > 0: time.sleep(config.ACTION_DELAY)

        raw_next_frame_bgr = self._capture_frame_raw_bgr()
        processed_next_frame_for_ai = self._preprocess_frame_for_ai(raw_next_frame_bgr)
        next_state_tensor = self._stack_frames_for_ai(processed_next_frame_for_ai)
        reward, done = self._get_reward_and_done(raw_next_frame_bgr)

        if config.ENABLE_GUI:
            with gui_data_lock:
                if config.GRAYSCALE:
                    gui_shared_data["ai_view"] = Image.fromarray(processed_next_frame_for_ai, 'L')
                else: 
                    gui_shared_data["ai_view"] = Image.fromarray(cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2RGB))
                if config.GUI_SHOW_RAW_CAPTURE:
                     gui_shared_data["raw_capture_view"] = Image.fromarray(cv2.cvtColor(raw_next_frame_bgr, cv2.COLOR_BGR2RGB))
        
        return next_state_tensor, reward, done, raw_next_frame_bgr

    def _match_template_cv(self, frame_area_to_search, template_img, threshold):
        if template_img is None: return False, 0.0
        if frame_area_to_search.shape[0] < template_img.shape[0] or \
           frame_area_to_search.shape[1] < template_img.shape[1]:
            return False, 0.0 
        
        result = cv2.matchTemplate(frame_area_to_search, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= threshold, max_val

    def _get_reward_and_done(self, current_raw_frame_bgr):
        done = False
        reward = config.REWARD_ALIVE

        frame_for_detection = current_raw_frame_bgr
        if config.GRAYSCALE and self.game_over_template is not None and len(self.game_over_template.shape)==2:
            frame_for_detection = cv2.cvtColor(current_raw_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.game_over_template is not None:
            search_area_go = frame_for_detection
            if config.GAME_OVER_SEARCH_REGION:
                x, y, w, h = config.GAME_OVER_SEARCH_REGION
                max_h, max_w = search_area_go.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, max_w - x), min(h, max_h - y)
                if w > 0 and h > 0: search_area_go = search_area_go[y:y+h, x:x+w]
            
            is_game_over, match_val = self._match_template_cv(search_area_go, self.game_over_template, config.GAME_OVER_DETECTION_THRESHOLD)
            if is_game_over:
                logging.debug(f"Game Over detected (match: {match_val:.2f}).")
                done = True
                reward = config.REWARD_DEATH
                if config.SAVE_FRAMES_ON_DEATH:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"death_frame_{ts}.png", current_raw_frame_bgr)
                return reward, done 
        
        
        if not done and config.REWARD_PROGRESS_FACTOR != 0:
            pass
        return reward, done

class Agent:
    def __init__(self, num_actions, sample_env_for_shape):
        self.num_actions = num_actions
        
        
        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, 
                              config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions,
                              config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.total_steps_done_in_training = 0

    def select_action(self, state_tensor):
        self.total_steps_done_in_training += 1
        current_epsilon = config.EPSILON_END + \
                          (config.EPSILON_START - config.EPSILON_END) * \
                          np.exp(-1. * self.total_steps_done_in_training / config.EPSILON_DECAY_FRAMES)
        
        action_val = 0
        q_values_list = [0.0] * self.num_actions

        if random.random() > current_epsilon:
            with torch.no_grad():
                q_values_tensor = self.policy_net(state_tensor)
                action_val = q_values_tensor.max(1)[1].item()
                q_values_list = q_values_tensor.cpu().squeeze().tolist()
                if not isinstance(q_values_list, list): q_values_list = [q_values_list] 
        else:
            action_val = random.randrange(self.num_actions)
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["q_values"] = q_values_list
                gui_shared_data["action"] = action_val
                gui_shared_data["epsilon"] = current_epsilon
        
        return torch.tensor([[action_val]], device=config.DEVICE, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or self.total_steps_done_in_training < config.LEARN_START_STEPS:
            return None
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=config.DEVICE, dtype=torch.bool)
        
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if not non_final_next_states_list: 
             non_final_next_states = None
        else:
            non_final_next_states = torch.cat(non_final_next_states_list)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.DEVICE)
        
        if non_final_next_states is not None and non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["current_loss"] = loss.item()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path=config.MODEL_SAVE_PATH):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps_done': self.total_steps_done_in_training,
            }, path)
            logging.info(f"Model saved: {path}")
        except Exception as e: logging.error(f"Error saving model: {e}")

    def load_model(self, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path): return False
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_done_in_training = checkpoint.get('total_steps_done', 0)
            
            self.policy_net.to(config.DEVICE)
            self.target_net.to(config.DEVICE)
            self.target_net.eval()
            for state_val in self.optimizer.state.values():
                for k, v_val in state_val.items():
                    if isinstance(v_val, torch.Tensor): state_val[k] = v_val.to(config.DEVICE)
            logging.info(f"Model loaded: {path}, Steps: {self.total_steps_done_in_training}")
            return True
        except Exception as e:
            logging.error(f"Error loading model {path}: {e}")
            return False
        
class AppGUI:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title(f"{config.PROJECT_NAME} Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self._on_gui_closing)
        self.root.geometry("900x650") 

        style = ttk.Style()
        style.theme_use('clam')

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
    
        vision_panel = ttk.Frame(main_frame)
        vision_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.ai_view_frame = ttk.LabelFrame(vision_panel, text="AI Processed View")
        self.ai_view_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.ai_view_label = ttk.Label(self.ai_view_frame)
        self.ai_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self._set_placeholder_image(self.ai_view_label, 
                                    config.FRAME_WIDTH * config.GUI_AI_VIEW_DISPLAY_SCALE, 
                                    config.FRAME_HEIGHT * config.GUI_AI_VIEW_DISPLAY_SCALE, "AI View")

        if config.GUI_SHOW_RAW_CAPTURE:
            self.raw_view_frame = ttk.LabelFrame(vision_panel, text="Raw Game Capture")
            self.raw_view_frame.pack(fill=tk.BOTH, expand=True)
            self.raw_view_label = ttk.Label(self.raw_view_frame)
            self.raw_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
            
            self._set_placeholder_image(self.raw_view_label, 
                                        int(800 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE), 
                                        int(600 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE), "Raw Capture")

        info_panel = ttk.Frame(main_frame)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        stats_frame = ttk.LabelFrame(info_panel, text="AI Statistics")
        stats_frame.pack(fill=tk.X, pady=(0,10))
        
        self.labels = {}
        info_order = [
            ("Episode", "Episode: 0 / 0"), ("Step", "Step: 0 / 0"),
            ("Total Steps", "Total Steps: 0"), ("Ep. Reward", "Ep. Reward: 0.00"),
            ("Avg Reward (100)", "Avg Reward (100): N/A"), ("Epsilon", "Epsilon: 1.0000"),
            ("Q-Values", "Q-Values: [0.00, 0.00]"),("Action", "Action: N/A"),
            ("Loss", "Loss: N/A"), ("Avg Loss (100)", "Avg Loss (100): N/A"),
            ("FPS", "FPS AI:0 | GUI:0"), ("Game Region", "Region: N/A")
        ]
        for key, text_val in info_order:
            self.labels[key] = ttk.Label(stats_frame, text=text_val, anchor="w")
            self.labels[key].pack(fill=tk.X, padx=5, pady=1)

        status_frame = ttk.LabelFrame(info_panel, text="Status")
        status_frame.pack(fill=tk.X, pady=(0,10))
        self.status_label = ttk.Label(status_frame, text="Initializing...", wraplength=280, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

        control_frame = ttk.LabelFrame(info_panel, text="Controls")
        control_frame.pack(fill=tk.X)
        self.pause_button = ttk.Button(control_frame, text="PAUSE (P)", command=self._toggle_pause)
        self.pause_button.pack(fill=tk.X, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="STOP AI", command=self._on_stop_button)
        self.stop_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.last_gui_update_time = time.perf_counter()
        self.gui_fps_counter = 0
        self.update_gui_info()


    def _set_placeholder_image(self, label_widget, width, height, text):
        placeholder = Image.new('RGB', (int(width), int(height)), color='gray')
        photo = ImageTk.PhotoImage(image=placeholder)
        label_widget.configure(image=photo)
        label_widget.image = photo 

    def _on_gui_closing(self):
        logging.info("GUI window closed by user.")
        stop_event.set()
        if pause_event.is_set(): pause_event.clear() 
        self.root.destroy()
        global gui_tk_root
        gui_tk_root = None 

    def _toggle_pause(self):
        if pause_event.is_set(): 
            pause_event.clear()
            self.pause_button.config(text=f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})")
            with gui_data_lock: gui_shared_data["is_paused"] = False
        else: 
            pause_event.set()
            self.pause_button.config(text=f"RESUME ({config.PAUSE_RESUME_KEY.upper()})")
            with gui_data_lock: gui_shared_data["is_paused"] = True
            
    def _on_stop_button(self):
        logging.info("STOP AI button pressed from GUI.")
        stop_event.set()
        if pause_event.is_set(): pause_event.clear()
        self.status_label.config(text="Stop signal sent. AI will halt.")
        self.stop_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)

    def update_gui_info(self):
        if not self.root or not self.root.winfo_exists(): return

        with gui_data_lock:
            data = gui_shared_data.copy()
            ai_view_img = data["ai_view"]
            raw_view_img = data["raw_capture_view"]

        if ai_view_img:
            w, h = ai_view_img.width, ai_view_img.height
            disp_w = int(w * config.GUI_AI_VIEW_DISPLAY_SCALE)
            disp_h = int(h * config.GUI_AI_VIEW_DISPLAY_SCALE)
            img_resized = ai_view_img.resize((disp_w, disp_h), Image.Resampling.NEAREST) 
            self.ai_view_photo = ImageTk.PhotoImage(image=img_resized)
            self.ai_view_label.configure(image=self.ai_view_photo)
            self.ai_view_label.image = self.ai_view_photo

        if config.GUI_SHOW_RAW_CAPTURE and raw_view_img:
            w, h = raw_view_img.width, raw_view_img.height
            disp_w = int(w * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            disp_h = int(h * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            img_resized = raw_view_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
            self.raw_view_photo = ImageTk.PhotoImage(image=img_resized)
            self.raw_view_label.configure(image=self.raw_view_photo)
            self.raw_view_label.image = self.raw_view_photo
            
        self.labels["Episode"].config(text=f"Episode: {data['episode']} / {config.NUM_EPISODES}")
        self.labels["Step"].config(text=f"Step: {data['step']} / {config.MAX_STEPS_PER_EPISODE}")
        self.labels["Total Steps"].config(text=f"Total Steps: {data['total_steps']}")
        self.labels["Ep. Reward"].config(text=f"Ep. Reward: {data['current_episode_reward']:.2f}")
        avg_r = sum(data['avg_reward_history']) / len(data['avg_reward_history']) if data['avg_reward_history'] else 'N/A'
        self.labels["Avg Reward (100)"].config(text=f"Avg Reward (100): {avg_r if isinstance(avg_r, str) else f'{avg_r:.2f}'}")
        self.labels["Epsilon"].config(text=f"Epsilon: {data['epsilon']:.4f}")
        q_str = ", ".join([f"{q:.2f}" for q in data['q_values']])
        self.labels["Q-Values"].config(text=f"Q-Values: [{q_str}]")
        action_str = "JUMP" if data['action'] == 1 else "IDLE"
        self.labels["Action"].config(text=f"Action: {action_str} ({data['action']})")
        self.labels["Loss"].config(text=f"Loss: {data['current_loss']:.4f}" if data['current_loss'] != 0 else "Loss: N/A")
        avg_l = sum(data['avg_loss_history']) / len(data['avg_loss_history']) if data['avg_loss_history'] else 'N/A'
        self.labels["Avg Loss (100)"].config(text=f"Avg Loss (100): {avg_l if isinstance(avg_l, str) else f'{avg_l:.4f}'}")
        self.labels["FPS"].config(text=data.get("fps_info", "AI: 0 | GUI: 0"))
        self.labels["Game Region"].config(text=data["game_region_info"])
        self.status_label.config(text=data["status_text"])
        
        self.pause_button.config(text=f"RESUME ({config.PAUSE_RESUME_KEY.upper()})" if data["is_paused"] else f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})")

        self.gui_fps_counter += 1
        current_time = time.perf_counter()
        if current_time - self.last_gui_update_time >= 1.0:
            actual_gui_fps = self.gui_fps_counter / (current_time - self.last_gui_update_time)
            self.last_gui_update_time = current_time
            self.gui_fps_counter = 0
            with gui_data_lock: 
                 current_ai_fps_info = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[0].strip()
                 gui_shared_data["fps_info"] = f"{current_ai_fps_info} | GUI: {actual_gui_fps:.1f}"
        
        if self.root and self.root.winfo_exists():
            self.root.after(config.GUI_UPDATE_INTERVAL_MS, self.update_gui_info)

def run_gui_in_thread():
    global gui_tk_root
    gui_tk_root = tk.Tk()
    app_gui = AppGUI(gui_tk_root)
    gui_tk_root.mainloop() 
    logging.info("GUI thread has finished.")
    if not stop_event.is_set(): 
        stop_event.set() 

def on_key_press(key):
    try:
        char_key = key.char
    except AttributeError:
        char_key = None 

    if char_key == config.PAUSE_RESUME_KEY:
        if gui_tk_root and gui_tk_root.winfo_exists(): 
            app_instance = None 
            
            
            if pause_event.is_set():
                pause_event.clear()
                with gui_data_lock: gui_shared_data["is_paused"] = False
                logging.info("AI Resumed via keyboard.")
            else:
                pause_event.set()
                with gui_data_lock: gui_shared_data["is_paused"] = True
                logging.info("AI Paused via keyboard.")
        else: 
            if pause_event.is_set(): pause_event.clear(); logging.info("AI Resumed (no GUI).")
            else: pause_event.set(); logging.info("AI Paused (no GUI).")


def plot_training_data(rewards, losses, durations, path=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        rewards_s = pd.Series(rewards)
        axs[0].plot(rewards_s, label='Ep. Reward', alpha=0.6)
        axs[0].plot(rewards_s.rolling(100, min_periods=1).mean(), label='Avg Reward (100ep)', color='red')
        axs[0].set_ylabel('Total Reward'); axs[0].legend(); axs[0].set_title('Rewards')

        valid_losses = [l for l in losses if l is not None]
        if valid_losses:
            losses_s = pd.Series(valid_losses)
            axs[1].plot(losses_s, label='Avg. Loss', alpha=0.6)
            axs[1].plot(losses_s.rolling(100, min_periods=1).mean(), label='Avg Loss (100ep)', color='green')
        axs[1].set_ylabel('Loss'); axs[1].legend(); axs[1].set_title('Loss')
        
        durations_s = pd.Series(durations)
        axs[2].plot(durations_s, label='Ep. Duration', alpha=0.6)
        axs[2].plot(durations_s.rolling(100, min_periods=1).mean(), label='Avg Duration (100ep)', color='purple')
        axs[2].set_xlabel('Episode'); axs[2].set_ylabel('Steps'); axs[2].legend(); axs[2].set_title('Durations')

        fig.suptitle(f'{config.PROJECT_NAME} Training Progress', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if path: plt.savefig(path); logging.info(f"Plots saved: {path}")
        plt.close(fig) 
    except ImportError: logging.warning("Matplotlib/Pandas not found. Skipping plots.")
    except Exception as e: logging.error(f"Plotting error: {e}")


def ai_training_main_loop():
    global gui_shared_data
    loop_start_time = time.perf_counter()
    ai_frames_this_second = 0
    last_ai_fps_update_time = loop_start_time

    if config.ENABLE_GUI:
        with gui_data_lock: gui_shared_data["status_text"] = "Initializing Environment & Agent..."
    
    env = GameEnvironment()
    if env.monitor_region is None and config.FALLBACK_GAME_REGION is None:
        logging.critical("FATAL: Game region could not be determined. Exiting AI thread.")
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = "ERROR: Game region undefined."
        stop_event.set()
        return

    agent = Agent(num_actions=config.NUM_ACTIONS, sample_env_for_shape=env)
    agent.load_model()

    all_ep_rewards, all_ep_avg_losses, all_ep_durations = [], [], []
    ai_loop_delay = 1.0 / config.AI_FPS_LIMIT if config.AI_FPS_LIMIT > 0 else 0
    total_step_counter_for_session = 0

    try:
        for i_episode in range(1, config.NUM_EPISODES + 1):
            if stop_event.is_set(): break
            
            if pause_event.is_set():
                logging.info("AI Training Paused...")
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["status_text"] = f"AI Paused (Press '{config.PAUSE_RESUME_KEY.upper()}' to resume)"
                pause_event.wait() 
                logging.info("AI Training Resumed.")
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["is_paused"] = False 

            current_state_tensor, _ = env.reset()
            current_episode_reward_val = 0.0
            episode_loss_sum = 0.0
            episode_opt_steps = 0
            
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["episode"] = i_episode
                    gui_shared_data["current_episode_reward"] = 0.0
                    gui_shared_data["status_text"] = f"Running Ep. {i_episode}..."

            for t_step in range(1, config.MAX_STEPS_PER_EPISODE + 1):
                if stop_event.is_set(): break
                if pause_event.is_set(): 
                    logging.info("AI Training Paused (mid-episode)...")
                    if config.ENABLE_GUI:
                         with gui_data_lock: gui_shared_data["status_text"] = f"AI Paused (Press '{config.PAUSE_RESUME_KEY.upper()}' to resume)"
                    pause_event.wait()
                    logging.info("AI Training Resumed (mid-episode).")
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["is_paused"] = False
                
                step_process_start_time = time.perf_counter()
                total_step_counter_for_session +=1
                
                action_tensor = agent.select_action(current_state_tensor)
                next_state_tensor, reward_val, done, _ = env.step(action_tensor.item())
                
                
                if config.REWARD_PROGRESS_FACTOR != 0 and not done:
                    progress = t_step / config.MAX_STEPS_PER_EPISODE
                    reward_val += config.REWARD_PROGRESS_FACTOR * progress

                current_episode_reward_val += reward_val
                
                agent.memory.push(current_state_tensor, action_tensor,
                                  None if done else next_state_tensor,
                                  torch.tensor([reward_val], device=config.DEVICE, dtype=torch.float),
                                  torch.tensor([done], device=config.DEVICE, dtype=torch.bool))
                current_state_tensor = next_state_tensor
                
                loss_item = agent.optimize_model()
                if loss_item is not None:
                    episode_loss_sum += loss_item
                    episode_opt_steps += 1

                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["step"] = t_step
                        gui_shared_data["total_steps"] = agent.total_steps_done_in_training 
                        gui_shared_data["current_episode_reward"] = current_episode_reward_val

                if ai_loop_delay > 0:
                    elapsed = time.perf_counter() - step_process_start_time
                    sleep_duration = ai_loop_delay - elapsed
                    if sleep_duration > 0: time.sleep(sleep_duration)
                
                ai_frames_this_second +=1
                current_time_fps = time.perf_counter()
                if current_time_fps - last_ai_fps_update_time >= 1.0:
                    actual_ai_fps = ai_frames_this_second / (current_time_fps - last_ai_fps_update_time)
                    last_ai_fps_update_time = current_time_fps
                    ai_frames_this_second = 0
                    if config.ENABLE_GUI:
                        with gui_data_lock:
                            current_gui_fps_info = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[-1].strip()
                            gui_shared_data["fps_info"] = f"AI: {actual_ai_fps:.1f} | {current_gui_fps_info}"

                if done: break            
            
            if stop_event.is_set(): break
            all_ep_rewards.append(current_episode_reward_val)
            all_ep_durations.append(t_step)
            avg_loss_this_ep = episode_loss_sum / episode_opt_steps if episode_opt_steps > 0 else None
            all_ep_avg_losses.append(avg_loss_this_ep)

            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["avg_reward_history"].append(current_episode_reward_val)
                    if avg_loss_this_ep is not None: gui_shared_data["avg_loss_history"].append(avg_loss_this_ep)
                    gui_shared_data["status_text"] = f"Ep {i_episode} Done. Reward: {current_episode_reward_val:.2f}"

            logging.info(f"Ep {i_episode}: Steps={t_step}, Reward={current_episode_reward_val:.2f}, AvgLoss={avg_loss_this_ep if avg_loss_this_ep is not None else 'N/A'}, Epsilon={gui_shared_data['epsilon']:.4f}, Mem={len(agent.memory)}")
            
            if i_episode % config.TARGET_UPDATE_FREQ_EPISODES == 0:
                agent.update_target_net()
                logging.info("Target network updated.")
            if i_episode % config.SAVE_MODEL_EVERY_N_EPISODES == 0:
                 agent.save_model()
                 if all_ep_rewards: plot_training_data(all_ep_rewards, all_ep_avg_losses, all_ep_durations)
        
    except KeyboardInterrupt: logging.info("Ctrl+C detected in AI loop. Stopping.")
    except Exception as e: logging.error(f"TRAINING LOOP CRASH: {e}", exc_info=True)
    finally:
        stop_event.set()
        logging.info("AI Training Loop Finalizing...")
        agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth")
        if all_ep_rewards: plot_training_data(all_ep_rewards, all_ep_avg_losses, all_ep_durations, f"{config.PROJECT_NAME.lower()}_final_plots.png")
        
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = "AI Training Finished. You can close GUI."
        logging.info("AI Training Loop Finished.")
        
        global game_region_display_window
        if game_region_display_window:
            game_region_display_window.destroy()
            game_region_display_window = None

if __name__ == '__main__':
    logging.info(f"--- {config.PROJECT_NAME} AI Starting --- PID: {os.getpid()}") 
    
    key_listener = pynput_keyboard.Listener(on_press=on_key_press)
    key_listener.start()
    logging.info(f"Keyboard listener started for '{config.PAUSE_RESUME_KEY}' key.")

    for i in range(2, 0, -1): 
        logging.info(f"Focus game window... Starting in {i}s")
        time.sleep(1)


    if config.ENABLE_GUI:
        gui_thread = threading.Thread(target=run_gui_in_thread, daemon=True)
        gui_thread.start()
        logging.info("GUI thread initiated.")
        time.sleep(0.5) 

    ai_thread = threading.Thread(target=ai_training_main_loop, daemon=True)
    ai_thread.start()
    logging.info("AI training thread initiated.")
    
    try:
        while ai_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.5) 
            
            if config.ENABLE_GUI and gui_tk_root is None and gui_thread.is_alive():
                logging.info("GUI window was closed. Signalling AI thread to stop.")
                stop_event.set() 
                break 
    except KeyboardInterrupt:
        logging.info("Ctrl+C in main thread. Initiating shutdown.")
        stop_event.set()
    
    logging.info("Waiting for AI thread to complete...")
    if pause_event.is_set(): pause_event.clear() 
    ai_thread.join(timeout=10)
    if ai_thread.is_alive():
        logging.warning("AI thread did not stop gracefully after 10s. Forcing.")

    if config.ENABLE_GUI and gui_thread.is_alive():
        logging.info("Waiting for GUI thread to complete...")
        if gui_tk_root and gui_tk_root.winfo_exists(): 
             gui_tk_root.destroy()
        gui_thread.join(timeout=5)
    
    key_listener.stop()
    logging.info("Keyboard listener stopped.")
    logging.info(f"--- {config.PROJECT_NAME} AI Shutdown Complete ---")
