
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
    
    for logger_name in ['matplotlib', 'PIL', 'pygetwindow', 'pynput']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.info(f"Logging for {config.PROJECT_NAME} setup. Level: {config.LOG_LEVEL.upper()}, Device: {config.DEVICE}")

setup_logging()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        logging.info(f"DQN Initialized: Input Channels={num_input_channels}, H={h}, W={w}, ConvOut H={final_h}, W={final_w}, LinearInputSize={linear_input_size}, Outputs={outputs}")

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
        self.game_window_handle = None
        
        self.last_window_check_time = 0

        self.monitor_region = self._update_and_focus_game_window(force_check=True)

        if self.monitor_region is None or self.monitor_region.get('width', 0) == 0:
            logging.error(f"Could not find or get valid dimensions for game window '{config.WINDOW_TITLE_SUBSTRING}'. Using fallback or first monitor.")
            self.monitor_region = config.FALLBACK_GAME_REGION if config.FALLBACK_GAME_REGION else self.sct.monitors[1]
        
        self._update_gui_game_region_info()
        logging.info(f"GameEnvironment Initialized. Screen region set to: {self.monitor_region}")
        
        self.game_over_screen_tpl = self._load_template(config.GAME_OVER_SCREEN_TEMPLATE_PATH, "Game Over Screen")
        self.player_death_effect_tpl = self._load_template(config.PLAYER_DEATH_EFFECT_TEMPLATE_PATH, "Player Death Effect")
        self.spike_tpl = self._load_template(config.SPIKE_TEMPLATE_PATH, "Spike")
        
        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)
        
        if config.SHOW_GAME_REGION_OUTLINE:
            self._create_region_display_window()

        self.sct = mss.mss()
        self.mouse = MouseController()
        self.game_window_handle = None 
        self.monitor_region = self._update_and_focus_game_window() 

        if self.monitor_region is None or self.monitor_region.get('width', 0) == 0:
            logging.error(f"Could not find or get valid dimensions for game window '{config.WINDOW_TITLE_SUBSTRING}'. Using fallback or first monitor.")
            self.monitor_region = config.FALLBACK_GAME_REGION if config.FALLBACK_GAME_REGION else self.sct.monitors[1]
        
        self._update_gui_game_region_info()
        logging.info(f"GameEnvironment Initialized. Screen region set to: {self.monitor_region}")
        
        self.game_over_screen_tpl = self._load_template(config.GAME_OVER_SCREEN_TEMPLATE_PATH, "Game Over Screen")
        self.player_death_effect_tpl = self._load_template(config.PLAYER_DEATH_EFFECT_TEMPLATE_PATH, "Player Death Effect")
        self.spike_tpl = self._load_template(config.SPIKE_TEMPLATE_PATH, "Spike")
        
        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)
        self.last_window_check_time = time.perf_counter()
        
        if config.SHOW_GAME_REGION_OUTLINE:
            self._create_region_display_window()

    def _update_gui_game_region_info(self):
        global gui_shared_data
        with gui_data_lock:
            gui_shared_data["game_region_info"] = f"L{self.monitor_region.get('left', 'N/A')}, T{self.monitor_region.get('top', 'N/A')}, W{self.monitor_region.get('width', 'N/A')}, H{self.monitor_region.get('height', 'N/A')}"

    def _load_template(self, path, template_name):
        if not path or not os.path.exists(path):
            logging.warning(f"{template_name} template path not specified or file not found: {path}")
            return None
        template_img = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
        if template_img is None:
            logging.error(f"Failed to load {template_name} template image from {path}")
            return None
        
        
        is_color_template_needed_for_ai = not config.GRAYSCALE
        
        if is_color_template_needed_for_ai:
            if len(template_img.shape) == 3 and template_img.shape[2] == 4: 
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2BGR)
            elif len(template_img.shape) == 2: 
                logging.warning(f"{template_name} template is grayscale, but AI is configured for color. This might lead to poor matching.")
        else: 
            if len(template_img.shape) == 3 and template_img.shape[2] == 4: 
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2GRAY)
            elif len(template_img.shape) == 3: 
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        logging.info(f"Loaded {template_name} template from: {path}, Final Shape: {template_img.shape}")
        return template_img

    def _update_and_focus_game_window(self, force_check=False):
        current_time = time.perf_counter()
        
        if not force_check and \
           (current_time - self.last_window_check_time < config.DYNAMIC_WINDOW_TRACKING_INTERVAL) and \
           self.game_window_handle and self.monitor_region.get('width',0) > 0 :
            return self.monitor_region

        self.last_window_check_time = current_time
        if not pygetwindow:
            logging.warning("pygetwindow not available, cannot auto-detect/track window.")
            return self.monitor_region 

        try:
            gd_windows = pygetwindow.getWindowsWithTitle(config.WINDOW_TITLE_SUBSTRING)
            if not gd_windows:
                self.game_window_handle = None
                logging.warning(f"No window with title substring '{config.WINDOW_TITLE_SUBSTRING}' found.")
                return self.monitor_region 
            
            gd_window = gd_windows[0]
            self.game_window_handle = getattr(gd_window, '_hWnd', None) 

            if gd_window.isMinimized:
                gd_window.restore()
                time.sleep(0.1) 

            if not gd_window.isActive:
                try:
                    gd_window.activate()
                    time.sleep(0.1) 
                except Exception as e_activate:
                    logging.debug(f"pygetwindow activate failed: {e_activate}")
                
                if Application and os.name == 'nt' and self.game_window_handle:
                    try:
                        app = Application().connect(handle=self.game_window_handle, timeout=1) 
                        app.window(handle=self.game_window_handle).set_focus()
                        logging.debug(f"Focused window '{gd_window.title}' using pywinauto.")
                    except Exception as e_pywinauto:
                        logging.debug(f"pywinauto focus failed: {e_pywinauto}")
            
            new_region = {
                "top": gd_window.top, "left": gd_window.left,
                "width": gd_window.width, "height": gd_window.height,
                "monitor": 1 
            }

            
            if new_region["width"] > 0 and new_region["height"] > 0:
                if self.monitor_region != new_region: 
                    logging.info(f"Game window moved/resized. Old: {self.monitor_region}, New: {new_region}")
                    self.monitor_region = new_region
                    self._update_gui_game_region_info()
                    if config.SHOW_GAME_REGION_OUTLINE:
                        self._update_region_display_window_geometry() 
                return self.monitor_region
            else:
                logging.warning(f"Detected window '{gd_window.title}' has invalid dimensions: W={new_region['width']}, H={new_region['height']}")
                self.game_window_handle = None 

        except Exception as e:
            logging.error(f"Error in _update_and_focus_game_window: {e}")
            self.game_window_handle = None 
        
        return self.monitor_region 

    def _create_region_display_window(self):
        global game_region_display_window_handle
        
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.DestroyWindow(game_region_display_window_handle)
            except Exception: pass 
            game_region_display_window_handle = None
        
        if not config.SHOW_GAME_REGION_OUTLINE or not win32gui: 
            return

        try:
            wc = win32gui.WNDCLASS()
            wc.hInstance = win32gui.GetModuleHandle(None)
            wc.lpszClassName = "GDintRegionFrameWindowClass" 
            wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH) 
            wc.lpfnWndProc = {win32con.WM_PAINT: self._on_paint_region_display} 
            
            class_atom = win32gui.RegisterClass(wc)
            
            dwExStyle = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
            hwnd = win32gui.CreateWindowEx(
                dwExStyle, class_atom, None, 
                win32con.WS_POPUP | win32con.WS_VISIBLE, 
                self.monitor_region['left'], self.monitor_region['top'], 
                self.monitor_region['width'], self.monitor_region['height'],
                None, None, wc.hInstance, None 
            )
            
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
            
            game_region_display_window_handle = hwnd
            logging.info("Game region outline display window (win32gui) created.")
        except Exception as e:
            logging.error(f"Failed to create win32gui region display window: {e}")
            game_region_display_window_handle = None

    def _on_paint_region_display(self, hwnd, msg, wparam, lparam):
        
        hdc, ps = win32gui.BeginPaint(hwnd)
        rect = win32gui.GetClientRect(hwnd)
        
        
        color_str = config.GAME_REGION_OUTLINE_BORDER_COLOR.lower()
        if color_str == "lime": color_gdi = 0x00FF00 
        elif color_str == "red": color_gdi = 0x0000FF
        elif color_str == "blue": color_gdi = 0xFF0000
        else: color_gdi = 0x00FF00 

        pen = win32gui.CreatePen(win32con.PS_SOLID, config.GAME_REGION_OUTLINE_THICKNESS, color_gdi)
        old_pen = win32gui.SelectObject(hdc, pen)
        
        
        null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
        old_brush = win32gui.SelectObject(hdc, null_brush)

        win32gui.Rectangle(hdc, 0, 0, rect[2], rect[3]) 
        
        win32gui.SelectObject(hdc, old_pen) 
        win32gui.SelectObject(hdc, old_brush) 
        win32gui.DeleteObject(pen) 
        
        win32gui.EndPaint(hwnd, ps)
        return 0 
        
    def _update_region_display_window_geometry(self):
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.SetWindowPos(game_region_display_window_handle, 
                                    win32con.HWND_TOPMOST, 
                                    self.monitor_region['left'], self.monitor_region['top'],
                                    self.monitor_region['width'], self.monitor_region['height'],
                                    win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW) 
                win32gui.InvalidateRect(game_region_display_window_handle, None, True) 
            except Exception as e:
                logging.warning(f"Could not update region display geometry/force repaint: {e}")

    def _capture_frame_raw_bgr(self):
        self._update_and_focus_game_window() 
        try:
            sct_img = self.sct.grab(self.monitor_region)
            
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        except mss.exception.ScreenShotError as e:
            logging.error(f"Screen capture error: {e}. Retrying after a short delay...")
            time.sleep(0.05) 
            return self._capture_frame_raw_bgr() 

    def _preprocess_frame_for_ai(self, frame_bgr):
        
        target_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if config.GRAYSCALE else frame_bgr
        
        return cv2.resize(target_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA).astype(np.uint8)

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

    def _detect_objects(self, frame_to_search_in, template_img, threshold):
        if template_img is None: return []
        
        
        if len(frame_to_search_in.shape) != len(template_img.shape) or \
           (len(template_img.shape) == 3 and frame_to_search_in.shape[2] != template_img.shape[2]):
            logging.debug(f"Skipping object detection due to channel mismatch: Frame {frame_to_search_in.shape}, Template {template_img.shape}")
            return []

        if frame_to_search_in.shape[0] < template_img.shape[0] or frame_to_search_in.shape[1] < template_img.shape[1]:
            return [] 

        detections = []
        result = cv2.matchTemplate(frame_to_search_in, template_img, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        template_h, template_w = template_img.shape[:2]

        
        for pt_y, pt_x in zip(locations[0], locations[1]): 
            detections.append((pt_x, pt_y, template_w, template_h))
        
        
        
        
        return detections[:config.MAX_SPIKES_TO_DRAW]


    def reset(self):
        self.stacked_frames.clear() 
        raw_frame_bgr = self._capture_frame_raw_bgr()
        processed_frame_for_ai = self._preprocess_frame_for_ai(raw_frame_bgr) 
        
        
        frame_chw = np.expand_dims(processed_frame_for_ai, axis=0) if config.GRAYSCALE else np.transpose(processed_frame_for_ai, (2,0,1))
        for _ in range(config.NUM_FRAMES_STACKED):
             self.stacked_frames.append(frame_chw)
            
        initial_stacked_state_tensor = torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float()
        
        return initial_stacked_state_tensor, raw_frame_bgr

    def step(self, action_value): 
        if action_value == 1: 
            self.mouse.press(Button.left)
            time.sleep(config.JUMP_DURATION)
            self.mouse.release(Button.left)
        
        if config.ACTION_DELAY > 0:
            time.sleep(config.ACTION_DELAY)

        raw_next_frame_bgr = self._capture_frame_raw_bgr()
        processed_next_frame_for_ai = self._preprocess_frame_for_ai(raw_next_frame_bgr)
        next_state_tensor = self._stack_frames_for_ai(processed_next_frame_for_ai) 

        reward, done = self._get_reward_and_done(raw_next_frame_bgr) 

        if config.ENABLE_GUI:
            detected_spikes_coords_in_ai_view = []
            if config.GUI_MARK_DETECTED_OBJECTS and self.spike_tpl is not None:
                
                frame_for_spike_search = processed_next_frame_for_ai 
                
                
                if not config.GRAYSCALE and len(self.spike_tpl.shape) == 2: 
                    frame_for_spike_search = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2GRAY)
                elif config.GRAYSCALE and len(self.spike_tpl.shape) == 3: 
                    
                    
                    pass 

                detected_spikes_coords_in_ai_view = self._detect_objects(
                    frame_for_spike_search, self.spike_tpl, config.SPIKE_DETECTION_THRESHOLD
                )
            
            with gui_data_lock:
                
                if config.GRAYSCALE: 
                    ai_view_gui_img = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_GRAY2RGB)
                else: 
                    ai_view_gui_img = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2RGB)
                gui_shared_data["ai_view"] = Image.fromarray(ai_view_gui_img)
                
                if config.GUI_SHOW_RAW_CAPTURE:
                    gui_shared_data["raw_capture_view"] = Image.fromarray(cv2.cvtColor(raw_next_frame_bgr, cv2.COLOR_BGR2RGB))
                
                gui_shared_data["detected_spikes"] = detected_spikes_coords_in_ai_view 
        
        return next_state_tensor, reward, done, raw_next_frame_bgr

    def _match_template_cv(self, frame_area_to_search, template_img, threshold):
        if template_img is None: return False, 0.0
        if frame_area_to_search.shape[0] < template_img.shape[0] or \
           frame_area_to_search.shape[1] < template_img.shape[1]:
            return False, 0.0 
        
        
        if len(frame_area_to_search.shape) != len(template_img.shape) or \
           (len(template_img.shape) == 3 and frame_area_to_search.shape[2] != template_img.shape[2]):
            
            
            logging.debug(f"Template matching type/channel mismatch. Frame: {frame_area_to_search.shape}, Template: {template_img.shape}")
            return False, 0.0
            
        result = cv2.matchTemplate(frame_area_to_search, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= threshold, max_val

    def _get_reward_and_done(self, current_raw_frame_bgr):
        done_flag = False
        current_reward = config.REWARD_ALIVE
        death_detected_this_step = False

        
        death_templates_to_check = []
        if self.game_over_screen_tpl:
            death_templates_to_check.append(
                (self.game_over_screen_tpl, config.GAME_OVER_SCREEN_DETECTION_THRESHOLD, "GameOverScreen")
            )
        if self.player_death_effect_tpl:
            death_templates_to_check.append(
                (self.player_death_effect_tpl, config.PLAYER_DEATH_EFFECT_THRESHOLD, "PlayerShatter")
            )

        for template_image, detection_threshold, template_name in death_templates_to_check:
            if template_image is None: continue

            
            frame_for_current_detection = current_raw_frame_bgr
            if len(template_image.shape) == 2: 
                if len(current_raw_frame_bgr.shape) == 3: 
                    frame_for_current_detection = cv2.cvtColor(current_raw_frame_bgr, cv2.COLOR_BGR2GRAY)
            

            search_area_for_template = frame_for_current_detection
            
            if config.GAME_OVER_SEARCH_REGION and template_name == "GameOverScreen":
                x, y, w, h = config.GAME_OVER_SEARCH_REGION
                max_h_frame, max_w_frame = search_area_for_template.shape[:2]
                
                x, y = max(0, x), max(0, y)
                w, h = min(w, max_w_frame - x), min(h, max_h_frame - y)
                if w > 0 and h > 0:
                    search_area_for_template = search_area_for_template[y:y+h, x:x+w]
            
            is_template_detected, match_value = self._match_template_cv(
                search_area_for_template, template_image, detection_threshold
            )
            if is_template_detected:
                logging.debug(f"Death detected by {template_name} (Match: {match_value:.2f} >= Threshold: {detection_threshold}).")
                death_detected_this_step = True
                break 
        
        if death_detected_this_step:
            done_flag = True
            current_reward = config.REWARD_DEATH
            if config.SAVE_FRAMES_ON_DEATH:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"death_frame_{timestamp_str}.png", current_raw_frame_bgr)
        
        return current_reward, done_flag


class Agent:
    def __init__(self, num_actions): 
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
        
        action_value_int = 0 
        q_values_for_gui = [0.0] * self.num_actions 

        if random.random() > current_epsilon: 
            with torch.no_grad():
                q_values_tensor = self.policy_net(state_tensor) 
                action_value_int = q_values_tensor.max(1)[1].item() 
                q_values_for_gui = q_values_tensor.cpu().squeeze().tolist()
                if not isinstance(q_values_for_gui, list): 
                    q_values_for_gui = [q_values_for_gui]
        else: 
            action_value_int = random.randrange(self.num_actions)
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["q_values"] = q_values_for_gui
                gui_shared_data["action"] = action_value_int
                gui_shared_data["epsilon"] = current_epsilon
        
        return torch.tensor([[action_value_int]], device=config.DEVICE, dtype=torch.long) 

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or \
           self.total_steps_done_in_training < config.LEARN_START_STEPS:
            return None 
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        
        
        batch = Transition(*zip(*transitions))

        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=config.DEVICE, dtype=torch.bool)
        
        
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if not non_final_next_states_list: 
             non_final_next_states_cat = None
        else:
            non_final_next_states_cat = torch.cat(non_final_next_states_list)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        
        
        
        
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.DEVICE)
        if non_final_next_states_cat is not None and non_final_next_states_cat.size(0) > 0: 
            with torch.no_grad(): 
                 next_state_values[non_final_mask] = self.target_net(non_final_next_states_cat).max(1)[0]
        
        
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) 

        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["current_loss"] = loss.item()
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
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Error saving model to {path}: {e}")

    def load_model(self, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path):
            logging.warning(f"Model file not found at {path}. Starting with a new model.")
            return False
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE) 
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict']) 
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_done_in_training = checkpoint.get('total_steps_done', 0) 
            
            
            self.policy_net.to(config.DEVICE)
            self.target_net.to(config.DEVICE)
            self.target_net.eval() 

            
            for state_value in self.optimizer.state.values():
                for k, v_value in state_value.items():
                    if isinstance(v_value, torch.Tensor):
                        state_value[k] = v_value.to(config.DEVICE)
            logging.info(f"Model loaded successfully from {path}. Resuming from {self.total_steps_done_in_training} total steps.")
            return True
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}. Starting with a new model.")
            return False


class AppGUI:
    def __init__(self, root_tk_instance):
        self.root = root_tk_instance
        self.root.title(f"{config.PROJECT_NAME} Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self._on_gui_closing) 
        self.root.geometry("950x700") 

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
        self._set_placeholder_image_for_label(self.ai_view_label, 
                                    config.FRAME_WIDTH * config.GUI_AI_VIEW_DISPLAY_SCALE, 
                                    config.FRAME_HEIGHT * config.GUI_AI_VIEW_DISPLAY_SCALE)

        if config.GUI_SHOW_RAW_CAPTURE:
            self.raw_view_frame = ttk.LabelFrame(vision_panel, text="Raw Game Capture")
            self.raw_view_frame.pack(fill=tk.BOTH, expand=True)
            self.raw_view_label = ttk.Label(self.raw_view_frame) 
            self.raw_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
            
            self._set_placeholder_image_for_label(self.raw_view_label, 
                                        int(800 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE), 
                                        int(600 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE))


        
        info_panel = ttk.Frame(main_frame)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0), ipadx=10) 
        
        stats_frame = ttk.LabelFrame(info_panel, text="AI Statistics")
        stats_frame.pack(fill=tk.X, pady=(0,10))
        
        self.info_labels = {} 
        
        label_definitions = [
            ("Episode", "Episode: 0 / 0"), ("Step", "Step: 0 / 0"),
            ("Total Steps", "Total Steps: 0"), ("Ep. Reward", "Ep. Reward: 0.00"),
            ("Avg Reward (100)", "Avg Reward (100): N/A"), ("Epsilon", "Epsilon: 1.0000"),
            ("Q-Values", "Q-Values: [0.00, 0.00]"),("Action", "Action: N/A"),
            ("Loss", "Loss: N/A"), ("Avg Loss (100)", "Avg Loss (100): N/A"),
            ("FPS", "FPS AI:0 | GUI:0"), ("Game Region", "Region: N/A")
        ]
        for key, default_text in label_definitions:
            self.info_labels[key] = ttk.Label(stats_frame, text=default_text, anchor="w")
            self.info_labels[key].pack(fill=tk.X, padx=5, pady=1)

        status_frame = ttk.LabelFrame(info_panel, text="Status")
        status_frame.pack(fill=tk.X, pady=(0,10))
        self.status_label = ttk.Label(status_frame, text="Initializing...", wraplength=300, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)

        control_frame = ttk.LabelFrame(info_panel, text="Controls")
        control_frame.pack(fill=tk.X)
        self.pause_button = ttk.Button(control_frame, text=f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})", command=self._toggle_pause_button_action)
        self.pause_button.pack(fill=tk.X, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="STOP AI", command=self._on_stop_button_press)
        self.stop_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.last_gui_update_timestamp = time.perf_counter()
        self.gui_frame_counter = 0
        self.update_gui_elements() 

    def _set_placeholder_image_for_label(self, label_widget, width, height):
        placeholder_img = Image.new('RGB', (int(width), int(height)), color='darkgrey')
        photo_img = ImageTk.PhotoImage(image=placeholder_img)
        label_widget.configure(image=photo_img)
        label_widget.image = photo_img 

    def _on_gui_closing(self):
        logging.info("GUI window closed by user.")
        stop_event.set() 
        if pause_event.is_set(): 
            pause_event.clear()
        self.root.destroy() 
        global gui_tk_root
        gui_tk_root = None 

    def _toggle_pause_button_action(self):
        if pause_event.is_set(): 
            pause_event.clear()
            logging.info("AI Resumed via GUI button.")
        else: 
            pause_event.set()
            logging.info("AI Paused via GUI button.")
        
        with gui_data_lock:
            gui_shared_data["is_paused"] = pause_event.is_set()
            
    def _on_stop_button_press(self):
        logging.info("STOP AI button pressed from GUI.")
        stop_event.set()
        if pause_event.is_set(): 
            pause_event.clear()
        self.status_label.config(text="Stop signal sent. AI will halt after current operation.")
        self.stop_button.config(state=tk.DISABLED) 
        self.pause_button.config(state=tk.DISABLED)

    def _draw_detections_on_pil_image(self, pil_image_obj, scaled_detections_list, color="red", line_thickness=1):
        if not scaled_detections_list:
            return pil_image_obj
        
        from PIL import ImageDraw 
        
        
        
        img_with_drawings = pil_image_obj
        draw_context = ImageDraw.Draw(img_with_drawings)
        
        for x_coord, y_coord, width_val, height_val in scaled_detections_list:
            draw_context.rectangle(
                [x_coord, y_coord, x_coord + width_val, y_coord + height_val],
                outline=color,
                width=line_thickness
            )
        return img_with_drawings

    def update_gui_elements(self):
        if not self.root or not self.root.winfo_exists(): 
            return

        with gui_data_lock: 
            current_data = gui_shared_data.copy()
            ai_processed_pil_image = current_data["ai_view"]
            raw_capture_pil_image = current_data["raw_capture_view"]
            current_detected_spikes = current_data.get("detected_spikes", []) 

        
        if ai_processed_pil_image:
            original_w, original_h = ai_processed_pil_image.width, ai_processed_pil_image.height
            display_w = int(original_w * config.GUI_AI_VIEW_DISPLAY_SCALE)
            display_h = int(original_h * config.GUI_AI_VIEW_DISPLAY_SCALE)
            
            
            displayable_ai_image = ai_processed_pil_image.resize((display_w, display_h), Image.Resampling.NEAREST) 

            if config.GUI_MARK_DETECTED_OBJECTS and current_detected_spikes:
                
                scaled_spike_coords = []
                for x, y, w, h in current_detected_spikes:
                    scaled_spike_coords.append((
                        int(x * config.GUI_AI_VIEW_DISPLAY_SCALE), int(y * config.GUI_AI_VIEW_DISPLAY_SCALE),
                        int(w * config.GUI_AI_VIEW_DISPLAY_SCALE), int(h * config.GUI_AI_VIEW_DISPLAY_SCALE)
                    ))
                
                displayable_ai_image = self._draw_detections_on_pil_image(displayable_ai_image, scaled_spike_coords, color="orange", line_thickness=2)

            self.ai_view_photo_tk = ImageTk.PhotoImage(image=displayable_ai_image)
            self.ai_view_label.configure(image=self.ai_view_photo_tk)
            self.ai_view_label.image = self.ai_view_photo_tk 

        
        if config.GUI_SHOW_RAW_CAPTURE and raw_capture_pil_image:
            original_w, original_h = raw_capture_pil_image.width, raw_capture_pil_image.height
            display_w = int(original_w * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            display_h = int(original_h * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            
            displayable_raw_image = raw_capture_pil_image.resize((display_w, display_h), Image.Resampling.LANCZOS) 
            self.raw_view_photo_tk = ImageTk.PhotoImage(image=displayable_raw_image)
            self.raw_view_label.configure(image=self.raw_view_photo_tk)
            self.raw_view_label.image = self.raw_view_photo_tk 
            
        
        self.info_labels["Episode"].config(text=f"Episode: {current_data['episode']} / {config.NUM_EPISODES}")
        self.info_labels["Step"].config(text=f"Step: {current_data['step']} / {config.MAX_STEPS_PER_EPISODE}")
        self.info_labels["Total Steps"].config(text=f"Total Steps: {current_data['total_steps']}")
        self.info_labels["Ep. Reward"].config(text=f"Ep. Reward: {current_data['current_episode_reward']:.2f}")
        avg_reward_val = sum(current_data['avg_reward_history']) / len(current_data['avg_reward_history']) if current_data['avg_reward_history'] else 'N/A'
        self.info_labels["Avg Reward (100)"].config(text=f"Avg Reward (100): {avg_reward_val if isinstance(avg_reward_val, str) else f'{avg_reward_val:.2f}'}")
        self.info_labels["Epsilon"].config(text=f"Epsilon: {current_data['epsilon']:.4f}")
        q_values_str = ", ".join([f"{q_val:.2f}" for q_val in current_data['q_values']])
        self.info_labels["Q-Values"].config(text=f"Q-Values: [{q_values_str}]")
        
        action_text = "JUMP" if current_data['action'] == 1 else "IDLE"
        
        action_display_color = "forest green" if current_data['action'] == 1 and len(current_data['q_values']) > 1 and current_data['q_values'][1] > current_data['q_values'][0] else "black"
        self.info_labels["Action"].config(text=f"Action: {action_text} ({current_data['action']})", foreground=action_display_color)
        
        self.info_labels["Loss"].config(text=f"Loss: {current_data['current_loss']:.4f}" if current_data['current_loss'] != 0 else "Loss: N/A")
        avg_loss_val = sum(current_data['avg_loss_history']) / len(current_data['avg_loss_history']) if current_data['avg_loss_history'] else 'N/A'
        self.info_labels["Avg Loss (100)"].config(text=f"Avg Loss (100): {avg_loss_val if isinstance(avg_loss_val, str) else f'{avg_loss_val:.4f}'}")
        
        self.info_labels["FPS"].config(text=current_data.get("fps_info", "AI: 0 | GUI: 0"))
        self.info_labels["Game Region"].config(text=current_data["game_region_info"])
        self.status_label.config(text=current_data["status_text"])
        
        
        self.pause_button.config(text=f"RESUME ({config.PAUSE_RESUME_KEY.upper()})" if current_data["is_paused"] else f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})")

        
        self.gui_frame_counter += 1
        current_timestamp = time.perf_counter()
        if current_timestamp - self.last_gui_update_timestamp >= 1.0: 
            actual_gui_fps = self.gui_frame_counter / (current_timestamp - self.last_gui_update_timestamp)
            self.last_gui_update_timestamp = current_timestamp
            self.gui_frame_counter = 0
            
            with gui_data_lock:
                 current_ai_fps_part = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[0].strip()
                 gui_shared_data["fps_info"] = f"{current_ai_fps_part} | GUI: {actual_gui_fps:.1f}"
        
        
        if self.root and self.root.winfo_exists(): 
            self.root.after(config.GUI_UPDATE_INTERVAL_MS, self.update_gui_elements)

def run_gui_in_thread():
    global gui_tk_root
    gui_tk_root = tk.Tk()
    app_gui_instance = AppGUI(gui_tk_root)
    gui_tk_root.mainloop() 
    logging.info("GUI thread has finished execution.")
    
    if not stop_event.is_set(): 
        stop_event.set()


def on_key_press_event(key):
    try:
        char_pressed = key.char
    except AttributeError: 
        return

    if char_pressed == config.PAUSE_RESUME_KEY:
        if pause_event.is_set(): 
            pause_event.clear() 
            logging.info("AI Resumed via keyboard press.")
        else: 
            pause_event.set() 
            logging.info("AI Paused via keyboard press.")
        
        with gui_data_lock:
            gui_shared_data["is_paused"] = pause_event.is_set()



def plot_training_data(episode_rewards_list, episode_losses_list, episode_durations_list, save_file_path=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd 
        plt.style.use('seaborn-v0_8-darkgrid') 

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True) 
        
        
        rewards_series = pd.Series(episode_rewards_list)
        axes[0].plot(rewards_series, label='Episode Reward', alpha=0.6)
        axes[0].plot(rewards_series.rolling(window=100, min_periods=1).mean(), label='Avg Reward (100 episodes)', color='red', linestyle='--')
        axes[0].set_ylabel('Total Reward')
        axes[0].legend()
        axes[0].set_title('Episode Rewards Over Time')

        
        valid_losses = [loss_val for loss_val in episode_losses_list if loss_val is not None]
        if valid_losses:
            losses_series = pd.Series(valid_losses)
            axes[1].plot(losses_series, label='Average Loss per Episode', alpha=0.6)
            axes[1].plot(losses_series.rolling(window=100, min_periods=1).mean(), label='Avg Loss (100 episodes)', color='green', linestyle='--')
        axes[1].set_ylabel('Average Loss')
        axes[1].legend()
        axes[1].set_title('Training Loss Over Time')
        
        
        durations_series = pd.Series(episode_durations_list)
        axes[2].plot(durations_series, label='Episode Duration (Steps)', alpha=0.6)
        axes[2].plot(durations_series.rolling(window=100, min_periods=1).mean(), label='Avg Duration (100 episodes)', color='purple', linestyle='--')
        axes[2].set_xlabel('Episode Number')
        axes[2].set_ylabel('Number of Steps')
        axes[2].legend()
        axes[2].set_title('Episode Durations Over Time')

        fig.suptitle(f'{config.PROJECT_NAME} Training Progress', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        
        if save_file_path:
            plt.savefig(save_file_path)
            logging.info(f"Training plots saved to {save_file_path}")
        plt.close(fig) 
    except ImportError:
        logging.warning("Matplotlib or Pandas not found. Skipping plot generation. Please install them to see plots.")
    except Exception as e:
        logging.error(f"An error occurred during plot generation: {e}")


def ai_training_main_loop():
    global gui_shared_data 
    global game_region_display_window_handle 

    loop_start_timestamp = time.perf_counter()
    ai_frames_processed_this_second = 0
    last_ai_fps_update_timestamp = loop_start_timestamp

    if config.ENABLE_GUI:
        with gui_data_lock:
            gui_shared_data["status_text"] = "Initializing Environment & AI Agent..."
    
    game_env = GameEnvironment() 
    
    if game_env.monitor_region is None or game_env.monitor_region.get('width', 0) == 0:
        logging.critical("FATAL ERROR: Game region could not be determined or is invalid. AI training cannot start.")
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["status_text"] = "ERROR: Game region undefined or invalid. Check logs."
        stop_event.set() 
        return

    ai_agent = Agent(num_actions=config.NUM_ACTIONS)
    ai_agent.load_model() 

    
    all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations = [], [], []
    
    
    ai_loop_target_delay = 1.0 / config.AI_FPS_LIMIT if config.AI_FPS_LIMIT > 0 else 0
    
    try:
        for episode_num in range(1, config.NUM_EPISODES + 1):
            if stop_event.is_set(): 
                logging.info("Stop event received in AI training loop. Terminating.")
                break
            
            if pause_event.is_set(): 
                logging.info("AI Training is Paused...")
                current_status_text = f"AI Training Paused (Press '{config.PAUSE_RESUME_KEY.upper()}' to resume)"
                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["status_text"] = current_status_text
                pause_event.wait() 
                logging.info("AI Training Resumed.")
                if config.ENABLE_GUI: 
                    with gui_data_lock:
                        gui_shared_data["is_paused"] = False 
                        gui_shared_data["status_text"] = f"Resuming Ep. {episode_num}..."


            current_state_tensor, _ = game_env.reset() 
            current_episode_total_reward = 0.0
            current_episode_loss_sum = 0.0
            current_episode_optimization_steps = 0
            
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["episode"] = episode_num
                    gui_shared_data["current_episode_reward"] = 0.0 
                    gui_shared_data["status_text"] = f"Running Episode {episode_num}..."

            for step_num_in_episode in range(1, config.MAX_STEPS_PER_EPISODE + 1):
                if stop_event.is_set(): break 
                if pause_event.is_set(): 
                    mid_ep_pause_status = f"AI Paused (Ep. {episode_num}, Step {step_num_in_episode})"
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["status_text"] = mid_ep_pause_status
                    pause_event.wait()
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["is_paused"] = False; gui_shared_data["status_text"] = f"Resuming Ep. {episode_num}..."

                
                step_processing_start_time = time.perf_counter()
                
                action_tensor = ai_agent.select_action(current_state_tensor) 
                next_state_tensor, reward_value, done_flag, _ = game_env.step(action_tensor.item()) 
                
                
                if config.REWARD_PROGRESS_FACTOR != 0 and not done_flag:
                    progress_reward = config.REWARD_PROGRESS_FACTOR * (step_num_in_episode / config.MAX_STEPS_PER_EPISODE)
                    reward_value += progress_reward

                current_episode_total_reward += reward_value
                
                
                ai_agent.memory.push(current_state_tensor, action_tensor,
                                  None if done_flag else next_state_tensor, 
                                  torch.tensor([reward_value], device=config.DEVICE, dtype=torch.float),
                                  torch.tensor([done_flag], device=config.DEVICE, dtype=torch.bool) 
                                  )
                current_state_tensor = next_state_tensor 
                
                
                loss_item_value = ai_agent.optimize_model()
                if loss_item_value is not None:
                    current_episode_loss_sum += loss_item_value
                    current_episode_optimization_steps += 1

                
                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["step"] = step_num_in_episode
                        gui_shared_data["total_steps"] = ai_agent.total_steps_done_in_training
                        gui_shared_data["current_episode_reward"] = current_episode_total_reward

                
                if ai_loop_target_delay > 0:
                    time_elapsed_this_step = time.perf_counter() - step_processing_start_time
                    sleep_duration_needed = ai_loop_target_delay - time_elapsed_this_step
                    if sleep_duration_needed > 0:
                        time.sleep(sleep_duration_needed)
                
                
                ai_frames_processed_this_second +=1
                current_loop_timestamp = time.perf_counter()
                if current_loop_timestamp - last_ai_fps_update_timestamp >= 1.0: 
                    actual_ai_fps_val = ai_frames_processed_this_second / (current_loop_timestamp - last_ai_fps_update_timestamp)
                    last_ai_fps_update_timestamp = current_loop_timestamp
                    ai_frames_processed_this_second = 0
                    if config.ENABLE_GUI: 
                        with gui_data_lock:
                            current_gui_fps_part = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[-1].strip() 
                            gui_shared_data["fps_info"] = f"AI: {actual_ai_fps_val:.1f} | {current_gui_fps_part}"

                if done_flag: 
                    break 
            
            
            if stop_event.is_set(): break 

            all_episode_total_rewards.append(current_episode_total_reward)
            all_episode_step_durations.append(step_num_in_episode) 
            avg_loss_this_episode = current_episode_loss_sum / current_episode_optimization_steps if current_episode_optimization_steps > 0 else None
            all_episode_avg_losses.append(avg_loss_this_episode)

            
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["avg_reward_history"].append(current_episode_total_reward)
                    if avg_loss_this_episode is not None:
                        gui_shared_data["avg_loss_history"].append(avg_loss_this_episode)
                    gui_shared_data["status_text"] = f"Episode {episode_num} Finished. Reward: {current_episode_total_reward:.2f}"

            logging.info(f"Episode {episode_num} Summary: Steps={step_num_in_episode}, Reward={current_episode_total_reward:.2f}, AvgLoss={avg_loss_this_episode if avg_loss_this_episode is not None else 'N/A'}, Epsilon={gui_shared_data['epsilon']:.4f}, MemorySize={len(ai_agent.memory)}")
            
            
            if episode_num % config.TARGET_UPDATE_FREQ_EPISODES == 0:
                ai_agent.update_target_net()
                logging.info("Target network updated.")
            
            
            if episode_num % config.SAVE_MODEL_EVERY_N_EPISODES == 0:
                 ai_agent.save_model()
                 if all_episode_total_rewards: 
                    plot_training_data(all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations)
        
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt (Ctrl+C) detected in AI training loop. Initiating stop.")
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR IN AI TRAINING LOOP: {e}", exc_info=True) 
    finally:
        stop_event.set() 
        logging.info("AI Training Loop finalizing procedures...")
        ai_agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth") 
        if all_episode_total_rewards: 
            plot_training_data(all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations, 
                               f"{config.PROJECT_NAME.lower()}_final_training_plots.png")
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["status_text"] = "AI Training Loop Finished. GUI can be closed."
        logging.info("AI Training Loop has finished.")
        
        
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.DestroyWindow(game_region_display_window_handle)
            except Exception: pass 
            game_region_display_window_handle = None


if __name__ == '__main__':
    logging.info(f"--- {config.PROJECT_NAME} AI Main Script Starting --- PID: {os.getpid()}")
    
    
    keyboard_listener_thread = pynput_keyboard.Listener(on_press=on_key_press_event)
    keyboard_listener_thread.start()
    logging.info(f"Keyboard listener started. Press '{config.PAUSE_RESUME_KEY}' to Pause/Resume AI.")

    
    for i in range(2, 0, -1): 
        logging.info(f"Please focus the Geometry Dash game window... Starting AI in {i} seconds.")
        time.sleep(1)

    
    gui_main_thread = None
    if config.ENABLE_GUI:
        gui_main_thread = threading.Thread(target=run_gui_in_thread, daemon=True) 
        gui_main_thread.start()
        logging.info("GUI thread has been initiated.")
        time.sleep(0.5) 

    
    ai_main_thread = threading.Thread(target=ai_training_main_loop, daemon=True) 
    ai_main_thread.start()
    logging.info("AI training thread has been initiated.")

    
    try:
        while ai_main_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.5) 
            
            if config.ENABLE_GUI and gui_tk_root is None and gui_main_thread and gui_main_thread.is_alive():
                logging.info("GUI window appears to have been closed. Signaling AI thread to stop.")
                stop_event.set() 
                break 
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected in main thread. Initiating graceful shutdown.")
        stop_event.set() 
    
    logging.info("Main thread: Waiting for AI training thread to complete...")
    if pause_event.is_set(): 
        pause_event.clear()
    ai_main_thread.join(timeout=10) 
    if ai_main_thread.is_alive():
        logging.warning("AI training thread did not stop gracefully within the timeout. It might be stuck.")

    if config.ENABLE_GUI and gui_main_thread and gui_main_thread.is_alive():
        logging.info("Main thread: Waiting for GUI thread to complete...")
        
        if gui_tk_root and gui_tk_root.winfo_exists():
             gui_tk_root.destroy()
        gui_main_thread.join(timeout=5) 
    
    keyboard_listener_thread.stop() 
    logging.info("Keyboard listener has been stopped.")
    logging.info(f"--- {config.PROJECT_NAME} AI Shutdown Sequence Complete ---")