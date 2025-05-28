# GDint.py
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
if os.name == 'nt':
    try:
        import win32gui
        import win32con
    except ImportError:
        win32gui = None
else:
    win32gui = None


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
    "is_paused": False, "detected_spikes": []
}
stop_event = threading.Event()
pause_event = threading.Event()
game_region_display_window_handle = None

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=log_format, handlers=[
        logging.FileHandler(config.LOG_FILE, mode='w'),
        logging.StreamHandler()
    ])
    for logger_name in ['matplotlib', 'PIL', 'pygetwindow', 'pynput']: logging.getLogger(logger_name).setLevel(logging.WARNING)
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
        def conv_out_size(size, kernel, stride, padding): return (size - kernel + 2 * padding) // stride + 1
        final_h = conv_out_size(conv_out_size(conv_out_size(h, 8, 4, 2), 4, 2, 1), 3, 1, 1)
        final_w = conv_out_size(conv_out_size(conv_out_size(w, 8, 4, 2), 4, 2, 1), 3, 1, 1)
        linear_input_size = final_h * final_w * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        logging.info(f"DQN: Input ({num_input_channels}, H:{h}, W:{w}), ConvOut ({final_h}, {final_w}), LinearIn: {linear_input_size}")
    def forward(self, x):
        x = x / 255.0; x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))); x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); return self.head(x)

class GameEnvironment:
    def __init__(self):
        self.sct = mss.mss()
        self.mouse = MouseController()
        self.game_window_handle = None
        self.monitor_region = self._update_and_focus_game_window()
        if self.monitor_region is None:
            logging.error(f"Window '{config.WINDOW_TITLE_SUBSTRING}' not found. Using fallback/first monitor.")
            self.monitor_region = config.FALLBACK_GAME_REGION if config.FALLBACK_GAME_REGION else self.sct.monitors[1]
        self._update_gui_game_region_info()
        logging.info(f"GameEnv: Screen region set to {self.monitor_region}")
        self.game_over_screen_tpl = self._load_template(config.GAME_OVER_SCREEN_TEMPLATE_PATH, "Game Over Screen")
        self.player_death_effect_tpl = self._load_template(config.PLAYER_DEATH_EFFECT_TEMPLATE_PATH, "Player Death Effect")
        self.spike_tpl = self._load_template(config.SPIKE_TEMPLATE_PATH, "Spike")
        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)
        self.last_window_check_time = time.perf_counter()
        if config.SHOW_GAME_REGION_OUTLINE: self._create_region_display_window()

    def _update_gui_game_region_info(self):
        global gui_shared_data
        with gui_data_lock:
            gui_shared_data["game_region_info"] = f"L{self.monitor_region['left']}, T{self.monitor_region['top']}, W{self.monitor_region['width']}, H{self.monitor_region['height']}"

    def _load_template(self, path, name):
        if not path or not os.path.exists(path):
            logging.warning(f"{name} template not specified/found: {path}"); return None
        tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if tpl is None: logging.error(f"Failed to load {name} template: {path}"); return None
        is_color_template_needed = not config.GRAYSCALE
        if is_color_template_needed:
            if len(tpl.shape) == 3 and tpl.shape[2] == 4: tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
            elif len(tpl.shape) == 2: logging.warning(f"{name} template is grayscale but color is configured for AI.")
        else: # Grayscale needed
            if len(tpl.shape) == 3 and tpl.shape[2] == 4: tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2GRAY)
            elif len(tpl.shape) == 3: tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        logging.info(f"Loaded {name} template: {path}, Shape: {tpl.shape}"); return tpl

    def _update_and_focus_game_window(self, force_check=False):
        current_time = time.perf_counter()
        if not force_check and current_time - self.last_window_check_time < config.DYNAMIC_WINDOW_TRACKING_INTERVAL:
            if self.game_window_handle: return self.monitor_region # Assume still valid
            # Fallthrough if handle is None
        self.last_window_check_time = current_time
        if not pygetwindow: return self.monitor_region # Return current/fallback if no lib
        try:
            gd_windows = pygetwindow.getWindowsWithTitle(config.WINDOW_TITLE_SUBSTRING)
            if not gd_windows: self.game_window_handle = None; return self.monitor_region
            gd_window = gd_windows[0]
            self.game_window_handle = gd_window._hWnd if hasattr(gd_window, '_hWnd') else None # For pywinauto

            if gd_window.isMinimized: gd_window.restore(); time.sleep(0.1)
            if not gd_window.isActive:
                try: gd_window.activate(); time.sleep(0.1)
                except Exception: pass # Best effort
                if Application and os.name == 'nt' and self.game_window_handle:
                    try:
                        app = Application().connect(handle=self.game_window_handle, timeout=1)
                        app.window(handle=self.game_window_handle).set_focus()
                    except Exception: pass # Best effort focus
            
            new_region = {"top": gd_window.top, "left": gd_window.left, "width": gd_window.width, "height": gd_window.height, "monitor": 1}
            if new_region["width"] > 0 and new_region["height"] > 0: # Basic sanity check
                if self.monitor_region != new_region:
                    logging.info(f"Game window moved/resized. Old: {self.monitor_region}, New: {new_region}")
                    self.monitor_region = new_region
                    self._update_gui_game_region_info()
                    if config.SHOW_GAME_REGION_OUTLINE: self._update_region_display_window_geometry()
                return self.monitor_region
        except Exception as e: logging.error(f"Error in _update_and_focus_game_window: {e}"); self.game_window_handle = None
        return self.monitor_region # Return old/fallback if error

    def _create_region_display_window(self):
        global game_region_display_window_handle
        if game_region_display_window_handle and win32gui:
            try: win32gui.DestroyWindow(game_region_display_window_handle); game_region_display_window_handle = None
            except Exception: pass
        if not win32gui or not config.SHOW_GAME_REGION_OUTLINE: return

        try:
            wc = win32gui.WNDCLASS()
            wc.hInstance = win32gui.GetModuleHandle(None)
            wc.lpszClassName = "GDintRegionFrame"
            wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH) # Transparent background
            wc.lpfnWndProc = {win32con.WM_PAINT: self._on_paint_region_display}
            classAtom = win32gui.RegisterClass(wc)
            
            dwExStyle = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
            hwnd = win32gui.CreateWindowEx(dwExStyle, classAtom, None, win32con.WS_POPUP | win32con.WS_VISIBLE,
                self.monitor_region['left'], self.monitor_region['top'], 
                self.monitor_region['width'], self.monitor_region['height'],
                None, None, wc.hInstance, None)
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA) # Full opacity for frame itself
            game_region_display_window_handle = hwnd
            logging.info("Game region outline display window created.")
        except Exception as e: logging.error(f"Failed to create region display (win32gui): {e}")

    def _on_paint_region_display(self, hwnd, msg, wparam, lparam):
        hdc, ps = win32gui.BeginPaint(hwnd)
        rect = win32gui.GetClientRect(hwnd)
        color = 0x00FF00 if config.GAME_REGION_OUTLINE_BORDER_COLOR == "lime" else 0x0000FF # BGR for GDI
        pen = win32gui.CreatePen(win32con.PS_SOLID, config.GAME_REGION_OUTLINE_THICKNESS, color)
        win32gui.SelectObject(hdc, pen)
        win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH)) # No fill
        win32gui.Rectangle(hdc, 0, 0, rect[2], rect[3])
        win32gui.DeleteObject(pen)
        win32gui.EndPaint(hwnd, ps)
        return 0
        
    def _update_region_display_window_geometry(self):
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.SetWindowPos(game_region_display_window_handle, win32con.HWND_TOPMOST,
                                    self.monitor_region['left'], self.monitor_region['top'],
                                    self.monitor_region['width'], self.monitor_region['height'],
                                    win32con.SWP_NOACTIVATE)
            except Exception as e: logging.warning(f"Could not update region display geometry: {e}")

    def _capture_frame_raw_bgr(self):
        self._update_and_focus_game_window() # Ensure region is current
        try:
            sct_img = self.sct.grab(self.monitor_region)
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        except mss.exception.ScreenShotError as e:
            logging.error(f"Screen capture error: {e}. Retrying..."); time.sleep(0.05)
            return self._capture_frame_raw_bgr()

    def _preprocess_frame_for_ai(self, frame_bgr):
        target_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if config.GRAYSCALE else frame_bgr
        return cv2.resize(target_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _stack_frames_for_ai(self, processed_frame_for_ai):
        frame_chw = np.expand_dims(processed_frame_for_ai, axis=0) if config.GRAYSCALE else np.transpose(processed_frame_for_ai, (2,0,1))
        if not self.stacked_frames: [self.stacked_frames.append(frame_chw) for _ in range(config.NUM_FRAMES_STACKED)]
        else: self.stacked_frames.append(frame_chw)
        return torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float()

    def _detect_objects(self, frame_to_search_in, template, threshold):
        if template is None: return []
        detections = []
        if frame_to_search_in.shape[0] < template.shape[0] or frame_to_search_in.shape[1] < template.shape[1]: return []
        res = cv2.matchTemplate(frame_to_search_in, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        th, tw = template.shape[:2]
        for pt in zip(*loc[::-1]): # Switch to (x, y)
            detections.append((pt[0], pt[1], tw, th)) # x, y, w, h
        # Non-maximum suppression could be added here if detections overlap heavily
        return detections[:config.MAX_SPIKES_TO_DRAW]


    def reset(self):
        self.stacked_frames.clear()
        raw_frame_bgr = self._capture_frame_raw_bgr()
        processed_frame = self._preprocess_frame_for_ai(raw_frame_bgr)
        frame_chw = np.expand_dims(processed_frame, axis=0) if config.GRAYSCALE else np.transpose(processed_frame, (2,0,1))
        for _ in range(config.NUM_FRAMES_STACKED): self.stacked_frames.append(frame_chw)
        return torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float(), raw_frame_bgr

    def step(self, action_value):
        if action_value == 1: self.mouse.press(Button.left); time.sleep(config.JUMP_DURATION); self.mouse.release(Button.left)
        if config.ACTION_DELAY > 0: time.sleep(config.ACTION_DELAY)
        raw_next_frame_bgr = self._capture_frame_raw_bgr()
        processed_next_frame_for_ai = self._preprocess_frame_for_ai(raw_next_frame_bgr)
        next_state_tensor = self._stack_frames_for_ai(processed_next_frame_for_ai)
        reward, done = self._get_reward_and_done(raw_next_frame_bgr)

        if config.ENABLE_GUI:
            detected_spikes_coords = []
            if config.GUI_MARK_DETECTED_OBJECTS and self.spike_tpl is not None:
                # Search for spikes in the AI's processed view
                frame_for_spike_detection = processed_next_frame_for_ai # This is already grayscale if config.GRAYSCALE is True
                if not config.GRAYSCALE and len(self.spike_tpl.shape) == 2: # Color AI view but grayscale spike template
                    frame_for_spike_detection = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2GRAY)

                detected_spikes_coords = self._detect_objects(frame_for_spike_detection, self.spike_tpl, config.SPIKE_DETECTION_THRESHOLD)
            
            with gui_data_lock:
                img_to_gui = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_GRAY2RGB) if config.GRAYSCALE else cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2RGB)
                gui_shared_data["ai_view"] = Image.fromarray(img_to_gui)
                if config.GUI_SHOW_RAW_CAPTURE: gui_shared_data["raw_capture_view"] = Image.fromarray(cv2.cvtColor(raw_next_frame_bgr, cv2.COLOR_BGR2RGB))
                gui_shared_data["detected_spikes"] = detected_spikes_coords
        return next_state_tensor, reward, done, raw_next_frame_bgr

    def _match_template_cv(self, frame_area, template, threshold):
        if template is None: return False, 0.0
        if frame_area.shape[0] < template.shape[0] or frame_area.shape[1] < template.shape[1]: return False, 0.0
        res = cv2.matchTemplate(frame_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res); return max_val >= threshold, max_val

    def _get_reward_and_done(self, current_raw_frame_bgr):
        done = False; reward = config.REWARD_ALIVE
        death_detected_by_any_template = False

        templates_to_check = []
        if self.game_over_screen_tpl: templates_to_check.append((self.game_over_screen_tpl, config.GAME_OVER_SCREEN_DETECTION_THRESHOLD, "ScreenTpl"))
        if self.player_death_effect_tpl: templates_to_check.append((self.player_death_effect_tpl, config.PLAYER_DEATH_EFFECT_THRESHOLD, "EffectTpl"))

        for tpl, threshold, tpl_name in templates_to_check:
            frame_for_detection = current_raw_frame_bgr
            # If template is grayscale but main detection frame isn't (and AI isn't grayscale), convert detection frame
            if len(tpl.shape) == 2 and (not config.GRAYSCALE or len(frame_for_detection.shape) == 3):
                frame_for_detection = cv2.cvtColor(current_raw_frame_bgr, cv2.COLOR_BGR2GRAY)
            elif len(tpl.shape) == 3 and len(frame_for_detection.shape) == 2 : # Color template, grayscale frame (unlikely if AI is color)
                 continue # Mismatch

            search_area = frame_for_detection
            if config.GAME_OVER_SEARCH_REGION and tpl_name == "ScreenTpl": # Only apply search region to screen template
                x,y,w,h = config.GAME_OVER_SEARCH_REGION; max_h, max_w = search_area.shape[:2]
                x,y=max(0,x),max(0,y); w,h=min(w,max_w-x),min(h,max_h-y)
                if w > 0 and h > 0: search_area = search_area[y:y+h, x:x+w]
            
            is_detected, match_val = self._match_template_cv(search_area, tpl, threshold)
            if is_detected:
                logging.debug(f"Death detected by {tpl_name} (match: {match_val:.2f})."); death_detected_by_any_template = True; break
        
        if death_detected_by_any_template:
            done = True; reward = config.REWARD_DEATH
            if config.SAVE_FRAMES_ON_DEATH:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); cv2.imwrite(f"death_frame_{ts}.png", current_raw_frame_bgr)
        return reward, done

class Agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE); self.total_steps_done_in_training = 0
    def select_action(self, state_tensor):
        self.total_steps_done_in_training += 1
        epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * np.exp(-1. * self.total_steps_done_in_training / config.EPSILON_DECAY_FRAMES)
        action_val = 0; q_vals_list = [0.0] * self.num_actions
        if random.random() > epsilon:
            with torch.no_grad(): q_tensor = self.policy_net(state_tensor); action_val = q_tensor.max(1)[1].item(); q_vals_list = q_tensor.cpu().squeeze().tolist()
            if not isinstance(q_vals_list, list): q_vals_list = [q_vals_list]
        else: action_val = random.randrange(self.num_actions)
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["q_values"] = q_vals_list; gui_shared_data["action"] = action_val; gui_shared_data["epsilon"] = epsilon
        return torch.tensor([[action_val]], device=config.DEVICE, dtype=torch.long)
    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or self.total_steps_done_in_training < config.LEARN_START_STEPS: return None
        transitions = self.memory.sample(config.BATCH_SIZE); batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.DEVICE, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(non_final_next_states_list) if non_final_next_states_list else None
        state_batch = torch.cat(batch.state); action_batch = torch.cat(batch.action); reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.DEVICE)
        if non_final_next_states is not None and non_final_next_states.size(0) > 0:
            with torch.no_grad(): next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); self.optimizer.step()
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["current_loss"] = loss.item()
        return loss.item()
    def update_target_net(self): self.target_net.load_state_dict(self.policy_net.state_dict())
    def save_model(self, path=config.MODEL_SAVE_PATH):
        try: torch.save({'policy_net_state_dict': self.policy_net.state_dict(), 'target_net_state_dict': self.target_net.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'total_steps_done': self.total_steps_done_in_training}, path); logging.info(f"Model saved: {path}")
        except Exception as e: logging.error(f"Error saving model: {e}")
    def load_model(self, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path): return False
        try:
            ckpt = torch.load(path, map_location=config.DEVICE); self.policy_net.load_state_dict(ckpt['policy_net_state_dict']); self.target_net.load_state_dict(ckpt['target_net_state_dict']); self.optimizer.load_state_dict(ckpt['optimizer_state_dict']); self.total_steps_done_in_training = ckpt.get('total_steps_done', 0)
            self.policy_net.to(config.DEVICE); self.target_net.to(config.DEVICE); self.target_net.eval()
            for state_val in self.optimizer.state.values():
                for k, v_val in state_val.items():
                    if isinstance(v_val, torch.Tensor): state_val[k] = v_val.to(config.DEVICE)
            logging.info(f"Model loaded: {path}, Steps: {self.total_steps_done_in_training}"); return True
        except Exception as e: logging.error(f"Error loading model {path}: {e}"); return False

class AppGUI:
    def __init__(self, root_tk):
        self.root = root_tk; self.root.title(f"{config.PROJECT_NAME} Dashboard"); self.root.protocol("WM_DELETE_WINDOW", self._on_gui_closing); self.root.geometry("950x700")
        style = ttk.Style(); style.theme_use('clam')
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        vision_panel = ttk.Frame(main_frame); vision_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.ai_view_frame = ttk.LabelFrame(vision_panel, text="AI Processed View"); self.ai_view_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.ai_view_label = ttk.Label(self.ai_view_frame); self.ai_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self._set_placeholder_image(self.ai_view_label, config.FRAME_WIDTH * config.GUI_AI_VIEW_DISPLAY_SCALE, config.FRAME_HEIGHT * config.GUI_AI_VIEW_DISPLAY_SCALE)
        if config.GUI_SHOW_RAW_CAPTURE:
            self.raw_view_frame = ttk.LabelFrame(vision_panel, text="Raw Game Capture"); self.raw_view_frame.pack(fill=tk.BOTH, expand=True)
            self.raw_view_label = ttk.Label(self.raw_view_frame); self.raw_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
            self._set_placeholder_image(self.raw_view_label, int(800*config.GUI_RAW_CAPTURE_DISPLAY_SCALE), int(600*config.GUI_RAW_CAPTURE_DISPLAY_SCALE))
        info_panel = ttk.Frame(main_frame); info_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0), ipadx=10)
        stats_frame = ttk.LabelFrame(info_panel, text="AI Statistics"); stats_frame.pack(fill=tk.X, pady=(0,10))
        self.labels = {}; info_order = [("Episode","Ep: 0/0"),("Step","St: 0/0"),("Total Steps","Tot St: 0"),("Ep. Reward","Ep Rwd: 0.00"),("Avg Reward (100)","Avg Rwd: N/A"),("Epsilon","Eps: 1.0000"),("Q-Values","Q: [0,0]"),("Action","Act: N/A"),("Loss","Loss: N/A"),("Avg Loss (100)","Avg Loss: N/A"),("FPS","FPS AI:0|GUI:0"),("Game Region","Reg: N/A")]
        for key,txt in info_order: self.labels[key]=ttk.Label(stats_frame,text=txt,anchor="w"); self.labels[key].pack(fill=tk.X,padx=5,pady=1)
        status_frame = ttk.LabelFrame(info_panel, text="Status"); status_frame.pack(fill=tk.X, pady=(0,10))
        self.status_label = ttk.Label(status_frame, text="Initializing...", wraplength=300, justify=tk.LEFT); self.status_label.pack(fill=tk.X, padx=5, pady=5)
        control_frame = ttk.LabelFrame(info_panel, text="Controls"); control_frame.pack(fill=tk.X)
        self.pause_button = ttk.Button(control_frame, text=f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})", command=self._toggle_pause); self.pause_button.pack(fill=tk.X, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="STOP AI", command=self._on_stop_button); self.stop_button.pack(fill=tk.X, padx=5, pady=5)
        self.last_gui_update_time = time.perf_counter(); self.gui_fps_counter = 0; self.update_gui_info()
    def _set_placeholder_image(self, lbl, w, h): ph=Image.new('RGB',(int(w),int(h)),color='gray'); pht=ImageTk.PhotoImage(image=ph); lbl.configure(image=pht); lbl.image=pht
    def _on_gui_closing(self): logging.info("GUI closed."); stop_event.set(); if pause_event.is_set(): pause_event.clear(); self.root.destroy(); global gui_tk_root; gui_tk_root = None
    def _toggle_pause(self):
        if pause_event.is_set(): pause_event.clear(); logging.info("AI Resumed via GUI.")
        else: pause_event.set(); logging.info("AI Paused via GUI.")
        with gui_data_lock: gui_shared_data["is_paused"] = pause_event.is_set()
    def _on_stop_button(self): logging.info("STOP AI GUI."); stop_event.set(); if pause_event.is_set(): pause_event.clear(); self.status_label.config(text="Stop signal sent."); self.stop_button.config(state=tk.DISABLED); self.pause_button.config(state=tk.DISABLED)
    def _draw_detections_on_pil_image(self, pil_img, detections_scaled, color="red", thickness=1):
        if not detections_scaled: return pil_img
        from PIL import ImageDraw
        draw = ImageDraw.Draw(pil_img)
        for x,y,w,h in detections_scaled: draw.rectangle([x,y, x+w, y+h], outline=color, width=thickness)
        return pil_img
    def update_gui_info(self):
        if not self.root or not self.root.winfo_exists(): return
        with gui_data_lock: data = gui_shared_data.copy(); ai_pil_img = data["ai_view"]; raw_pil_img = data["raw_capture_view"]; detected_spikes = data.get("detected_spikes", [])
        if ai_pil_img:
            w,h = ai_pil_img.width, ai_pil_img.height; disp_w, disp_h = int(w*config.GUI_AI_VIEW_DISPLAY_SCALE), int(h*config.GUI_AI_VIEW_DISPLAY_SCALE)
            if config.GUI_MARK_DETECTED_OBJECTS and detected_spikes:
                # Scale spike coords to display size
                scaled_spikes = [(int(x*config.GUI_AI_VIEW_DISPLAY_SCALE), int(y*config.GUI_AI_VIEW_DISPLAY_SCALE), int(sw*config.GUI_AI_VIEW_DISPLAY_SCALE), int(sh*config.GUI_AI_VIEW_DISPLAY_SCALE)) for x,y,sw,sh in detected_spikes]
                # Create a mutable copy for drawing
                displayable_ai_img = ai_pil_img.copy().resize((disp_w,disp_h), Image.Resampling.NEAREST)
                displayable_ai_img = self._draw_detections_on_pil_image(displayable_ai_img, scaled_spikes, color="orange", thickness=2)
            else: displayable_ai_img = ai_pil_img.resize((disp_w,disp_h), Image.Resampling.NEAREAST)
            self.ai_view_photo = ImageTk.PhotoImage(image=displayable_ai_img); self.ai_view_label.configure(image=self.ai_view_photo); self.ai_view_label.image = self.ai_view_photo
        if config.GUI_SHOW_RAW_CAPTURE and raw_pil_img:
            w,h = raw_pil_img.width,raw_pil_img.height; disp_w, disp_h = int(w*config.GUI_RAW_CAPTURE_DISPLAY_SCALE), int(h*config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            img_res = raw_pil_img.resize((disp_w,disp_h), Image.Resampling.LANCZOS)
            self.raw_view_photo = ImageTk.PhotoImage(image=img_res); self.raw_view_label.configure(image=self.raw_view_photo); self.raw_view_label.image = self.raw_view_photo
        self.labels["Episode"].config(text=f"Ep: {data['episode']}/{config.NUM_EPISODES}"); self.labels["Step"].config(text=f"St: {data['step']}/{config.MAX_STEPS_PER_EPISODE}"); self.labels["Total Steps"].config(text=f"Tot St: {data['total_steps']}")
        self.labels["Ep. Reward"].config(text=f"Ep Rwd: {data['current_episode_reward']:.2f}"); avg_r = sum(data['avg_reward_history'])/len(data['avg_reward_history']) if data['avg_reward_history'] else 'N/A'; self.labels["Avg Reward (100)"].config(text=f"Avg Rwd: {avg_r if isinstance(avg_r,str) else f'{avg_r:.2f}'}")
        self.labels["Epsilon"].config(text=f"Eps: {data['epsilon']:.4f}"); q_s = ", ".join([f"{q:.2f}" for q in data['q_values']]); self.labels["Q-Values"].config(text=f"Q: [{q_s}]")
        action_str = "JUMP" if data['action']==1 else "IDLE"; action_color = "green" if data['action']==1 and data['q_values'][1] > data['q_values'][0] else "black"
        self.labels["Action"].config(text=f"Act: {action_str} ({data['action']})", foreground=action_color)
        self.labels["Loss"].config(text=f"Loss: {data['current_loss']:.4f}" if data['current_loss']!=0 else "Loss: N/A"); avg_l = sum(data['avg_loss_history'])/len(data['avg_loss_history']) if data['avg_loss_history'] else 'N/A'; self.labels["Avg Loss (100)"].config(text=f"Avg Loss: {avg_l if isinstance(avg_l,str) else f'{avg_l:.4f}'}")
        self.labels["FPS"].config(text=data.get("fps_info","AI:0|GUI:0")); self.labels["Game Region"].config(text=data["game_region_info"]); self.status_label.config(text=data["status_text"])
        self.pause_button.config(text=f"RESUME ({config.PAUSE_RESUME_KEY.upper()})" if data["is_paused"] else f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})")
        self.gui_fps_counter+=1; cur_tm = time.perf_counter()
        if cur_tm - self.last_gui_update_time >= 1.0:
            act_gui_fps=self.gui_fps_counter/(cur_tm-self.last_gui_update_time); self.last_gui_update_time=cur_tm; self.gui_fps_counter=0
            with gui_data_lock: cur_ai_fps_info = gui_shared_data.get("fps_info","AI:0|GUI:0").split("|")[0].strip(); gui_shared_data["fps_info"]=f"{cur_ai_fps_info}|GUI:{act_gui_fps:.1f}"
        if self.root and self.root.winfo_exists(): self.root.after(config.GUI_UPDATE_INTERVAL_MS, self.update_gui_info)

def run_gui_in_thread(): global gui_tk_root; gui_tk_root = tk.Tk(); AppGUI(gui_tk_root); gui_tk_root.mainloop(); logging.info("GUI thread finished."); stop_event.set() # Signal AI if GUI closes
def on_key_press(key):
    try: char_key = key.char
    except AttributeError: return
    if char_key == config.PAUSE_RESUME_KEY:
        if pause_event.is_set(): pause_event.clear(); logging.info("AI Resumed (kbd).")
        else: pause_event.set(); logging.info("AI Paused (kbd).")
        with gui_data_lock: gui_shared_data["is_paused"] = pause_event.is_set()
def plot_training_data(r,l,d,p=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt; import pandas as pd; plt.style.use('seaborn-v0_8-darkgrid'); fig,axs=plt.subplots(3,1,figsize=(12,15),sharex=True)
        rs=pd.Series(r);axs[0].plot(rs,label='Ep.R',alpha=0.6);axs[0].plot(rs.rolling(100,min_periods=1).mean(),label='AvgR(100)',color='r');axs[0].set_ylabel('Reward');axs[0].legend();axs[0].set_title('Rewards')
        vl=[x for x in l if x is not None];
        if vl: ls=pd.Series(vl);axs[1].plot(ls,label='Avg.L',alpha=0.6);axs[1].plot(ls.rolling(100,min_periods=1).mean(),label='AvgL(100)',color='g');
        axs[1].set_ylabel('Loss');axs[1].legend();axs[1].set_title('Loss')
        ds=pd.Series(d);axs[2].plot(ds,label='Ep.Dur',alpha=0.6);axs[2].plot(ds.rolling(100,min_periods=1).mean(),label='AvgDur(100)',color='purple');axs[2].set_xlabel('Episode');axs[2].set_ylabel('Steps');axs[2].legend();axs[2].set_title('Durations')
        fig.suptitle(f'{config.PROJECT_NAME} Stats',fontsize=16);plt.tight_layout(rect=[0,0.03,1,0.95]);
        if p: plt.savefig(p); logging.info(f"Plots: {p}"); plt.close(fig)
    except Exception as e: logging.warning(f"Plotting error: {e}")

def ai_training_main_loop():
    global gui_shared_data, game_region_display_window_handle
    loop_st_tm=time.perf_counter(); ai_fr_sec=0; last_ai_fps_upd=loop_st_tm
    if config.ENABLE_GUI:
        with gui_data_lock: gui_shared_data["status_text"] = "Init Env & Agent..."
    env = GameEnvironment(); agent = Agent(num_actions=config.NUM_ACTIONS)
    if env.monitor_region is None or env.monitor_region['width'] == 0: logging.critical("FATAL: Game region invalid. Exit AI."); stop_event.set(); return
    agent.load_model()
    all_r,all_l,all_d = [],[],[]
    ai_delay = 1.0/config.AI_FPS_LIMIT if config.AI_FPS_LIMIT > 0 else 0
    try:
        for i_ep in range(1, config.NUM_EPISODES+1):
            if stop_event.is_set(): break
            if pause_event.is_set():
                logging.info("AI Paused..."); status_txt = f"AI Paused (Press '{config.PAUSE_RESUME_KEY.upper()}')"
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["status_text"] = status_txt
                pause_event.wait(); logging.info("AI Resumed.")
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["is_paused"] = False
            
            cur_state_t, _ = env.reset(); ep_rwd=0.0; ep_l_sum=0.0; ep_opt_st=0
            if config.ENABLE_GUI:
                with gui_data_lock: gui_shared_data["episode"]=i_ep; gui_shared_data["current_episode_reward"]=0.0; gui_shared_data["status_text"]=f"Running Ep.{i_ep}..."
            for t_st in range(1, config.MAX_STEPS_PER_EPISODE+1):
                if stop_event.is_set(): break
                if pause_event.is_set(): # Mid-episode pause check
                    status_txt_mid = f"AI Paused (Press '{config.PAUSE_RESUME_KEY.upper()}')"
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["status_text"] = status_txt_mid
                    pause_event.wait()
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["is_paused"] = False
                
                step_proc_st_tm = time.perf_counter()
                action_t = agent.select_action(cur_state_t)
                next_state_t, rwd_val, done, _ = env.step(action_t.item())
                if config.REWARD_PROGRESS_FACTOR != 0 and not done: rwd_val += config.REWARD_PROGRESS_FACTOR * (t_st/config.MAX_STEPS_PER_EPISODE)
                ep_rwd += rwd_val
                agent.memory.push(cur_state_t,action_t, None if done else next_state_t, torch.tensor([rwd_val],device=config.DEVICE,dtype=torch.float), torch.tensor([done],device=config.DEVICE,dtype=torch.bool))
                cur_state_t = next_state_t
                loss_i = agent.optimize_model()
                if loss_i is not None: ep_l_sum+=loss_i; ep_opt_st+=1
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["step"]=t_st; gui_shared_data["total_steps"]=agent.total_steps_done_in_training; gui_shared_data["current_episode_reward"]=ep_rwd
                if ai_delay > 0: elpsd = time.perf_counter()-step_proc_st_tm; slp_dur = ai_delay - elpsd; time.sleep(slp_dur) if slp_dur > 0 else None
                ai_fr_sec+=1; cur_tm_fps = time.perf_counter()
                if cur_tm_fps - last_ai_fps_upd >= 1.0:
                    act_ai_fps=ai_fr_sec/(cur_tm_fps-last_ai_fps_upd); last_ai_fps_upd=cur_tm_fps; ai_fr_sec=0
                    if config.ENABLE_GUI:
                        with gui_data_lock: cur_gui_fps_info=gui_shared_data.get("fps_info","AI:0|GUI:0").split("|")[-1].strip(); gui_shared_data["fps_info"]=f"AI:{act_ai_fps:.1f}|{cur_gui_fps_info}"
                if done: break
            if stop_event.is_set(): break
            all_r.append(ep_rwd); all_d.append(t_st); avg_l_ep = ep_l_sum/ep_opt_st if ep_opt_st > 0 else None; all_l.append(avg_l_ep)
            if config.ENABLE_GUI:
                with gui_data_lock: gui_shared_data["avg_reward_history"].append(ep_rwd); \
                                    (gui_shared_data["avg_loss_history"].append(avg_l_ep) if avg_l_ep is not None else None); \
                                    gui_shared_data["status_text"]=f"Ep {i_ep} Done. Rwd: {ep_rwd:.2f}"
            logging.info(f"Ep {i_ep}: St={t_st}, Rwd={ep_rwd:.2f}, AvgL={avg_l_ep if avg_l_ep is not None else 'N/A'}, Eps={gui_shared_data['epsilon']:.4f}, Mem={len(agent.memory)}")
            if i_ep % config.TARGET_UPDATE_FREQ_EPISODES == 0: agent.update_target_net(); logging.info("Target net updated.")
            if i_ep % config.SAVE_MODEL_EVERY_N_EPISODES == 0: agent.save_model(); plot_training_data(all_r,all_l,all_d) if all_r else None
    except KeyboardInterrupt: logging.info("Ctrl+C in AI loop. Stopping.")
    except Exception as e: logging.error(f"TRAINING LOOP CRASH: {e}", exc_info=True)
    finally:
        stop_event.set(); logging.info("AI Loop Finalizing..."); agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth")
        plot_training_data(all_r,all_l,all_d, f"{config.PROJECT_NAME.lower()}_final_plots.png") if all_r else None
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"]="AI Training Finished."
        logging.info("AI Loop Finished.")
        if game_region_display_window_handle and win32gui:
            try: win32gui.DestroyWindow(game_region_display_window_handle); game_region_display_window_handle = None
            except Exception: pass

if __name__ == '__main__':
    logging.info(f"--- {config.PROJECT_NAME} AI Starting --- PID: {os.getpid()}"); key_listener = pynput_keyboard.Listener(on_press=on_key_press); key_listener.start(); logging.info(f"Kbd listener for '{config.PAUSE_RESUME_KEY}'.")
    [logging.info(f"Focus game... Start in {i}s") or time.sleep(1) for i in range(2,0,-1)]
    gui_thread = threading.Thread(target=run_gui_in_thread,daemon=True) if config.ENABLE_GUI else None
    if gui_thread: gui_thread.start(); logging.info("GUI thread started."); time.sleep(0.5)
    ai_thread = threading.Thread(target=ai_training_main_loop,daemon=True); ai_thread.start(); logging.info("AI training thread started.")
    try:
        while ai_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.5)
            if config.ENABLE_GUI and gui_tk_root is None and gui_thread and gui_thread.is_alive(): stop_event.set(); break # GUI closed
    except KeyboardInterrupt: logging.info("Ctrl+C in main. Shutdown."); stop_event.set()
    logging.info("Waiting AI thread..."); (pause_event.clear() if pause_event.is_set() else None); ai_thread.join(timeout=10)
    if config.ENABLE_GUI and gui_thread and gui_thread.is_alive():
        logging.info("Waiting GUI thread..."); (gui_tk_root.destroy() if gui_tk_root and gui_tk_root.winfo_exists() else None) ; gui_thread.join(timeout=5)
    key_listener.stop(); logging.info(f"--- {config.PROJECT_NAME} Shutdown Complete ---")