#Вызов духа здравого смысла. Он просит тебя: "Комментируй, сука"
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

# Для Windows-специфичных функций отображения рамки
if os.name == 'nt':
    try:
        import win32gui
        import win32con
    except ImportError:
        win32gui = None # Устанавливаем в None, если импорт не удался
else:
    win32gui = None # Не Windows, win32gui не применимо


import GDint_config as config

# --- Global Variables & Events ---
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
pause_event = threading.Event() # False = running, True = paused
game_region_display_window_handle = None # Для Windows HWND

# --- Logging Setup ---
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=log_format, handlers=[
        logging.FileHandler(config.LOG_FILE, mode='w'),
        logging.StreamHandler()
    ])
    # Уменьшаем "болтливость" сторонних библиотек
    for logger_name in ['matplotlib', 'PIL', 'pygetwindow', 'pynput']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.info(f"Logging for {config.PROJECT_NAME} setup. Level: {config.LOG_LEVEL.upper()}, Device: {config.DEVICE}")

setup_logging()

# --- Helper Classes ---
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

# --- DQN Model ---
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
        x = x / 255.0 # Normalize
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.head(x)

# --- Game Environment ---
class GameEnvironment:
    def __init__(self):
        self.sct = mss.mss()
        self.mouse = MouseController()
        self.game_window_handle = None # Store HWND for pywinauto/win32gui if needed
        self.monitor_region = self._update_and_focus_game_window() # Initial detection

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
        template_img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # Load with alpha if present
        if template_img is None:
            logging.error(f"Failed to load {template_name} template image from {path}")
            return None
        
        # Handle color/grayscale conversion based on config and template type
        is_color_template_needed_for_ai = not config.GRAYSCALE
        
        if is_color_template_needed_for_ai:
            if len(template_img.shape) == 3 and template_img.shape[2] == 4: # BGRA
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2BGR)
            elif len(template_img.shape) == 2: # Grayscale template but color AI
                logging.warning(f"{template_name} template is grayscale, but AI is configured for color. This might lead to poor matching.")
        else: # Grayscale AI
            if len(template_img.shape) == 3 and template_img.shape[2] == 4: # BGRA
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2GRAY)
            elif len(template_img.shape) == 3: # BGR
                template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        logging.info(f"Loaded {template_name} template from: {path}, Final Shape: {template_img.shape}")
        return template_img

    def _update_and_focus_game_window(self, force_check=False):
        current_time = time.perf_counter()
        # Only check if interval passed or forced, or if we don't have a valid window handle/region
        if not force_check and \
           (current_time - self.last_window_check_time < config.DYNAMIC_WINDOW_TRACKING_INTERVAL) and \
           self.game_window_handle and self.monitor_region.get('width',0) > 0 :
            return self.monitor_region

        self.last_window_check_time = current_time
        if not pygetwindow:
            logging.warning("pygetwindow not available, cannot auto-detect/track window.")
            return self.monitor_region # Return current region (might be fallback)

        try:
            gd_windows = pygetwindow.getWindowsWithTitle(config.WINDOW_TITLE_SUBSTRING)
            if not gd_windows:
                self.game_window_handle = None
                logging.warning(f"No window with title substring '{config.WINDOW_TITLE_SUBSTRING}' found.")
                return self.monitor_region # Return current (possibly fallback)
            
            gd_window = gd_windows[0]
            self.game_window_handle = getattr(gd_window, '_hWnd', None) # For pywinauto/win32gui

            if gd_window.isMinimized:
                gd_window.restore()
                time.sleep(0.1) # Allow window to restore

            if not gd_window.isActive:
                try:
                    gd_window.activate()
                    time.sleep(0.1) # Allow activation
                except Exception as e_activate:
                    logging.debug(f"pygetwindow activate failed: {e_activate}")
                # Try with pywinauto if on Windows and available
                if Application and os.name == 'nt' and self.game_window_handle:
                    try:
                        app = Application().connect(handle=self.game_window_handle, timeout=1) # Short timeout
                        app.window(handle=self.game_window_handle).set_focus()
                        logging.debug(f"Focused window '{gd_window.title}' using pywinauto.")
                    except Exception as e_pywinauto:
                        logging.debug(f"pywinauto focus failed: {e_pywinauto}")
            
            new_region = {
                "top": gd_window.top, "left": gd_window.left,
                "width": gd_window.width, "height": gd_window.height,
                "monitor": 1 # mss requires a monitor, assume primary for simplicity
            }

            # Basic sanity check for valid dimensions
            if new_region["width"] > 0 and new_region["height"] > 0:
                if self.monitor_region != new_region: # Check if it actually changed
                    logging.info(f"Game window moved/resized. Old: {self.monitor_region}, New: {new_region}")
                    self.monitor_region = new_region
                    self._update_gui_game_region_info()
                    if config.SHOW_GAME_REGION_OUTLINE:
                        self._update_region_display_window_geometry() # Update outline window
                return self.monitor_region
            else:
                logging.warning(f"Detected window '{gd_window.title}' has invalid dimensions: W={new_region['width']}, H={new_region['height']}")
                self.game_window_handle = None # Invalidate handle if dimensions are bad

        except Exception as e:
            logging.error(f"Error in _update_and_focus_game_window: {e}")
            self.game_window_handle = None # Invalidate on error
        
        return self.monitor_region # Return current or fallback region if anything failed

    def _create_region_display_window(self):
        global game_region_display_window_handle
        # Clean up existing window if any
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.DestroyWindow(game_region_display_window_handle)
            except Exception: pass # Ignore if already destroyed
            game_region_display_window_handle = None
        
        if not config.SHOW_GAME_REGION_OUTLINE or not win32gui: # Only on Windows with win32gui
            return

        try:
            wc = win32gui.WNDCLASS()
            wc.hInstance = win32gui.GetModuleHandle(None)
            wc.lpszClassName = "GDintRegionFrameWindowClass" # Unique class name
            wc.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
            wc.hbrBackground = win32gui.GetStockObject(win32con.NULL_BRUSH) # Makes background transparent for drawing
            wc.lpfnWndProc = {win32con.WM_PAINT: self._on_paint_region_display} # Message map
            
            class_atom = win32gui.RegisterClass(wc)
            
            dwExStyle = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW
            hwnd = win32gui.CreateWindowEx(
                dwExStyle, class_atom, None, # Window class, No title
                win32con.WS_POPUP | win32con.WS_VISIBLE, # Styles
                self.monitor_region['left'], self.monitor_region['top'], 
                self.monitor_region['width'], self.monitor_region['height'],
                None, None, wc.hInstance, None # Parent, Menu, Instance, lParam
            )
            # Set full opacity for the window itself, transparency is handled by NULL_BRUSH and WS_EX_LAYERED
            win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
            
            game_region_display_window_handle = hwnd
            logging.info("Game region outline display window (win32gui) created.")
        except Exception as e:
            logging.error(f"Failed to create win32gui region display window: {e}")
            game_region_display_window_handle = None

    def _on_paint_region_display(self, hwnd, msg, wparam, lparam):
        # This is the WM_PAINT handler for the win32gui overlay window
        hdc, ps = win32gui.BeginPaint(hwnd)
        rect = win32gui.GetClientRect(hwnd)
        
        # Convert color string to BGR for GDI
        color_str = config.GAME_REGION_OUTLINE_BORDER_COLOR.lower()
        if color_str == "lime": color_gdi = 0x00FF00 # GDI uses BGR: 0xBBGGRR
        elif color_str == "red": color_gdi = 0x0000FF
        elif color_str == "blue": color_gdi = 0xFF0000
        else: color_gdi = 0x00FF00 # Default to lime

        pen = win32gui.CreatePen(win32con.PS_SOLID, config.GAME_REGION_OUTLINE_THICKNESS, color_gdi)
        old_pen = win32gui.SelectObject(hdc, pen)
        
        # Use NULL_BRUSH to make the rectangle fill transparent
        null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
        old_brush = win32gui.SelectObject(hdc, null_brush)

        win32gui.Rectangle(hdc, 0, 0, rect[2], rect[3]) # Draw rectangle border
        
        win32gui.SelectObject(hdc, old_pen) # Restore old pen
        win32gui.SelectObject(hdc, old_brush) # Restore old brush
        win32gui.DeleteObject(pen) # Delete created pen
        
        win32gui.EndPaint(hwnd, ps)
        return 0 # Must return 0 for WM_PAINT
        
    def _update_region_display_window_geometry(self):
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.SetWindowPos(game_region_display_window_handle, 
                                    win32con.HWND_TOPMOST, # Keep it on top
                                    self.monitor_region['left'], self.monitor_region['top'],
                                    self.monitor_region['width'], self.monitor_region['height'],
                                    win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW) # Don't activate it, ensure visible
                win32gui.InvalidateRect(game_region_display_window_handle, None, True) # Force repaint
            except Exception as e:
                logging.warning(f"Could not update region display geometry/force repaint: {e}")

    def _capture_frame_raw_bgr(self):
        self._update_and_focus_game_window() # Ensure region and focus are current
        try:
            sct_img = self.sct.grab(self.monitor_region)
            # Convert BGRA (from mss) to BGR
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        except mss.exception.ScreenShotError as e:
            logging.error(f"Screen capture error: {e}. Retrying after a short delay...")
            time.sleep(0.05) # Small delay before retry
            return self._capture_frame_raw_bgr() # Recursive call

    def _preprocess_frame_for_ai(self, frame_bgr):
        # Convert to grayscale if configured
        target_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if config.GRAYSCALE else frame_bgr
        # Resize to the dimensions expected by the AI model
        return cv2.resize(target_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA).astype(np.uint8)

    def _stack_frames_for_ai(self, processed_frame_for_ai): # Expects (H, W) or (H, W, C)
        # Ensure frame has channel dimension for stacking: (1, H, W) or (C, H, W)
        if config.GRAYSCALE: # processed_frame_for_ai is (H, W)
            frame_chw = np.expand_dims(processed_frame_for_ai, axis=0) # -> (1, H, W)
        else: # processed_frame_for_ai is (H, W, C)
            frame_chw = np.transpose(processed_frame_for_ai, (2, 0, 1)) # -> (C, H, W)

        # Initialize deque if it's empty
        if not self.stacked_frames:
            for _ in range(config.NUM_FRAMES_STACKED):
                self.stacked_frames.append(frame_chw)
        else:
            self.stacked_frames.append(frame_chw) # Add new frame, oldest is removed by deque
        
        # Concatenate frames along the channel axis
        stacked_state_tensor_data = np.concatenate(list(self.stacked_frames), axis=0) # Results in (N*C, H, W)
        # Add batch dimension and convert to float tensor for PyTorch
        return torch.from_numpy(stacked_state_tensor_data).unsqueeze(0).to(config.DEVICE).float()

    def _detect_objects(self, frame_to_search_in, template_img, threshold):
        if template_img is None: return []
        # Ensure frame_to_search_in and template_img have compatible types/channels for matchTemplate
        # This check is basic; more robust handling might be needed if templates vary wildly
        if len(frame_to_search_in.shape) != len(template_img.shape) or \
           (len(template_img.shape) == 3 and frame_to_search_in.shape[2] != template_img.shape[2]):
            logging.debug(f"Skipping object detection due to channel mismatch: Frame {frame_to_search_in.shape}, Template {template_img.shape}")
            return []

        if frame_to_search_in.shape[0] < template_img.shape[0] or frame_to_search_in.shape[1] < template_img.shape[1]:
            return [] # Template larger than search area

        detections = []
        result = cv2.matchTemplate(frame_to_search_in, template_img, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        template_h, template_w = template_img.shape[:2]

        # Store as (x, y, w, h)
        for pt_y, pt_x in zip(locations[0], locations[1]): # loc is (y_coords, x_coords)
            detections.append((pt_x, pt_y, template_w, template_h))
        
        # Basic Non-Maximum Suppression (NMS) - can be improved
        # This simple version just takes the first few if MAX_SPIKES_TO_DRAW is small
        # For more robust NMS, a proper algorithm should be used if many overlaps occur.
        return detections[:config.MAX_SPIKES_TO_DRAW]


    def reset(self):
        self.stacked_frames.clear() # Clear previous frames
        raw_frame_bgr = self._capture_frame_raw_bgr()
        processed_frame_for_ai = self._preprocess_frame_for_ai(raw_frame_bgr) # (H,W) or (H,W,C)
        
        # Initialize stack with the first frame
        frame_chw = np.expand_dims(processed_frame_for_ai, axis=0) if config.GRAYSCALE else np.transpose(processed_frame_for_ai, (2,0,1))
        for _ in range(config.NUM_FRAMES_STACKED):
             self.stacked_frames.append(frame_chw)
            
        initial_stacked_state_tensor = torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float()
        
        return initial_stacked_state_tensor, raw_frame_bgr

    def step(self, action_value): # action_value is an int (0 or 1)
        if action_value == 1: # Jump
            self.mouse.press(Button.left)
            time.sleep(config.JUMP_DURATION)
            self.mouse.release(Button.left)
        
        if config.ACTION_DELAY > 0:
            time.sleep(config.ACTION_DELAY)

        raw_next_frame_bgr = self._capture_frame_raw_bgr()
        processed_next_frame_for_ai = self._preprocess_frame_for_ai(raw_next_frame_bgr)
        next_state_tensor = self._stack_frames_for_ai(processed_next_frame_for_ai) # Adds new frame and stacks

        reward, done = self._get_reward_and_done(raw_next_frame_bgr) # Use the latest raw frame for death detection

        if config.ENABLE_GUI:
            detected_spikes_coords_in_ai_view = []
            if config.GUI_MARK_DETECTED_OBJECTS and self.spike_tpl is not None:
                # Frame for spike detection should match spike_tpl's color type
                frame_for_spike_search = processed_next_frame_for_ai # This is already (H,W) if GRAYSCALE, or (H,W,C) if color
                
                # If AI is color but spike template is grayscale (or vice-versa and needs conversion)
                if not config.GRAYSCALE and len(self.spike_tpl.shape) == 2: # AI Color, Spike Gray
                    frame_for_spike_search = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2GRAY)
                elif config.GRAYSCALE and len(self.spike_tpl.shape) == 3: # AI Gray, Spike Color (less likely to be intended)
                    # This case is tricky. Forcing spike_tpl to gray during load is better.
                    # If spike_tpl is color, and frame_for_spike_search is gray, matchTemplate will fail.
                    pass # Assuming spike_tpl matches config.GRAYSCALE or is handled at load

                detected_spikes_coords_in_ai_view = self._detect_objects(
                    frame_for_spike_search, self.spike_tpl, config.SPIKE_DETECTION_THRESHOLD
                )
            
            with gui_data_lock:
                # Prepare AI view for GUI (convert to RGB PIL Image)
                if config.GRAYSCALE: # processed_next_frame_for_ai is (H,W)
                    ai_view_gui_img = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_GRAY2RGB)
                else: # processed_next_frame_for_ai is (H,W,C) BGR
                    ai_view_gui_img = cv2.cvtColor(processed_next_frame_for_ai, cv2.COLOR_BGR2RGB)
                gui_shared_data["ai_view"] = Image.fromarray(ai_view_gui_img)
                
                if config.GUI_SHOW_RAW_CAPTURE:
                    gui_shared_data["raw_capture_view"] = Image.fromarray(cv2.cvtColor(raw_next_frame_bgr, cv2.COLOR_BGR2RGB))
                
                gui_shared_data["detected_spikes"] = detected_spikes_coords_in_ai_view # These are in AI view's coordinate space
        
        return next_state_tensor, reward, done, raw_next_frame_bgr

    def _match_template_cv(self, frame_area_to_search, template_img, threshold):
        if template_img is None: return False, 0.0
        if frame_area_to_search.shape[0] < template_img.shape[0] or \
           frame_area_to_search.shape[1] < template_img.shape[1]:
            return False, 0.0 # Template larger than search area
        
        # Ensure consistent types for matchTemplate
        if len(frame_area_to_search.shape) != len(template_img.shape) or \
           (len(template_img.shape) == 3 and frame_area_to_search.shape[2] != template_img.shape[2]):
            # This can happen if one is color and other is gray. Attempt conversion if feasible.
            # However, templates should ideally be pre-processed to match expected frame type.
            logging.debug(f"Template matching type/channel mismatch. Frame: {frame_area_to_search.shape}, Template: {template_img.shape}")
            return False, 0.0
            
        result = cv2.matchTemplate(frame_area_to_search, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= threshold, max_val

    def _get_reward_and_done(self, current_raw_frame_bgr):
        done_flag = False
        current_reward = config.REWARD_ALIVE
        death_detected_this_step = False

        # List of (template, threshold, name) tuples
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

            # Determine the frame to use for detection based on template's color type
            frame_for_current_detection = current_raw_frame_bgr
            if len(template_image.shape) == 2: # Template is grayscale
                if len(current_raw_frame_bgr.shape) == 3: # Raw frame is color
                    frame_for_current_detection = cv2.cvtColor(current_raw_frame_bgr, cv2.COLOR_BGR2GRAY)
            # else: template is color, use raw_frame_bgr as is (assuming it's BGR)

            search_area_for_template = frame_for_current_detection
            # Apply search region only if specified and it's the generic "Game Over" screen template
            if config.GAME_OVER_SEARCH_REGION and template_name == "GameOverScreen":
                x, y, w, h = config.GAME_OVER_SEARCH_REGION
                max_h_frame, max_w_frame = search_area_for_template.shape[:2]
                # Ensure search region is within frame bounds
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
                break # Stop checking other death templates if one is found
        
        if death_detected_this_step:
            done_flag = True
            current_reward = config.REWARD_DEATH
            if config.SAVE_FRAMES_ON_DEATH:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"death_frame_{timestamp_str}.png", current_raw_frame_bgr)
        
        return current_reward, done_flag

# --- Agent ---
class Agent:
    def __init__(self, num_actions): # Removed sample_env_for_shape
        self.num_actions = num_actions
        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, 
                              config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions,
                              config.NUM_FRAMES_STACKED, config.GRAYSCALE).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.total_steps_done_in_training = 0 # Tracks steps for epsilon decay and learning start

    def select_action(self, state_tensor): # state_tensor is (1, C_total, H, W)
        self.total_steps_done_in_training += 1
        current_epsilon = config.EPSILON_END + \
                          (config.EPSILON_START - config.EPSILON_END) * \
                          np.exp(-1. * self.total_steps_done_in_training / config.EPSILON_DECAY_FRAMES)
        
        action_value_int = 0 # Default to 0 (e.g., do nothing)
        q_values_for_gui = [0.0] * self.num_actions # Initialize for GUI

        if random.random() > current_epsilon: # Exploit: choose best action
            with torch.no_grad():
                q_values_tensor = self.policy_net(state_tensor) # Output shape: (1, num_actions)
                action_value_int = q_values_tensor.max(1)[1].item() # Get index of max Q-value
                q_values_for_gui = q_values_tensor.cpu().squeeze().tolist()
                if not isinstance(q_values_for_gui, list): # Ensure it's a list even for single action
                    q_values_for_gui = [q_values_for_gui]
        else: # Explore: choose random action
            action_value_int = random.randrange(self.num_actions)
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["q_values"] = q_values_for_gui
                gui_shared_data["action"] = action_value_int
                gui_shared_data["epsilon"] = current_epsilon
        
        return torch.tensor([[action_value_int]], device=config.DEVICE, dtype=torch.long) # Return as tensor

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or \
           self.total_steps_done_in_training < config.LEARN_START_STEPS:
            return None # Not enough samples or not yet time to learn
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=config.DEVICE, dtype=torch.bool)
        
        # Concatenate only non-None next_states
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if not non_final_next_states_list: # All next states in batch are None (terminal)
             non_final_next_states_cat = None
        else:
            non_final_next_states_cat = torch.cat(non_final_next_states_list)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a): model computes Q(s_t), then we select the columns of actions taken.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have V(s) = 0 if s is a final state.
        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.DEVICE)
        if non_final_next_states_cat is not None and non_final_next_states_cat.size(0) > 0: # Check if there are any non-final states
            with torch.no_grad(): # No gradients for target_net
                 next_state_values[non_final_mask] = self.target_net(non_final_next_states_cat).max(1)[0]
        
        # Compute the expected Q values (Bellman equation)
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

        # Compute Huber loss (Smooth L1 Loss)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # Unsqueeze target to match shape

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Clip gradients to prevent explosion
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
                'target_net_state_dict': self.target_net.state_dict(), # Save target too for consistency if needed
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
            checkpoint = torch.load(path, map_location=config.DEVICE) # Load to specified device
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict']) # Load target from checkpoint
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_done_in_training = checkpoint.get('total_steps_done', 0) # .get for backward compatibility
            
            # Ensure models are on the correct device after loading
            self.policy_net.to(config.DEVICE)
            self.target_net.to(config.DEVICE)
            self.target_net.eval() # Target network is always in evaluation mode

            # Move optimizer states to the correct device
            for state_value in self.optimizer.state.values():
                for k, v_value in state_value.items():
                    if isinstance(v_value, torch.Tensor):
                        state_value[k] = v_value.to(config.DEVICE)
            logging.info(f"Model loaded successfully from {path}. Resuming from {self.total_steps_done_in_training} total steps.")
            return True
        except Exception as e:
            logging.error(f"Error loading model from {path}: {e}. Starting with a new model.")
            return False

# --- GUI Application ---
class AppGUI:
    def __init__(self, root_tk_instance):
        self.root = root_tk_instance
        self.root.title(f"{config.PROJECT_NAME} Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self._on_gui_closing) # Handle window close button
        self.root.geometry("950x700") # Adjusted size for potentially more info

        style = ttk.Style()
        style.theme_use('clam') # A decent theme

        # Main layout frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel: Vision (AI View and Raw Capture)
        vision_panel = ttk.Frame(main_frame)
        vision_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.ai_view_frame = ttk.LabelFrame(vision_panel, text="AI Processed View")
        self.ai_view_frame.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        self.ai_view_label = ttk.Label(self.ai_view_frame) # Will hold the image
        self.ai_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self._set_placeholder_image_for_label(self.ai_view_label, 
                                    config.FRAME_WIDTH * config.GUI_AI_VIEW_DISPLAY_SCALE, 
                                    config.FRAME_HEIGHT * config.GUI_AI_VIEW_DISPLAY_SCALE)

        if config.GUI_SHOW_RAW_CAPTURE:
            self.raw_view_frame = ttk.LabelFrame(vision_panel, text="Raw Game Capture")
            self.raw_view_frame.pack(fill=tk.BOTH, expand=True)
            self.raw_view_label = ttk.Label(self.raw_view_frame) # Will hold the image
            self.raw_view_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
            # Placeholder size for raw capture can be an estimate or updated dynamically
            self._set_placeholder_image_for_label(self.raw_view_label, 
                                        int(800 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE), # Example placeholder size
                                        int(600 * config.GUI_RAW_CAPTURE_DISPLAY_SCALE))


        # Right Panel: Info & Controls
        info_panel = ttk.Frame(main_frame)
        info_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0), ipadx=10) # ipadx for internal padding
        
        stats_frame = ttk.LabelFrame(info_panel, text="AI Statistics")
        stats_frame.pack(fill=tk.X, pady=(0,10))
        
        self.info_labels = {} # Dictionary to store all dynamic labels
        # Define order and default text for labels
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
        self.update_gui_elements() # Start the update loop

    def _set_placeholder_image_for_label(self, label_widget, width, height):
        placeholder_img = Image.new('RGB', (int(width), int(height)), color='darkgrey')
        photo_img = ImageTk.PhotoImage(image=placeholder_img)
        label_widget.configure(image=photo_img)
        label_widget.image = photo_img # Keep a reference to prevent garbage collection

    def _on_gui_closing(self):
        logging.info("GUI window closed by user.")
        stop_event.set() # Signal main AI loop to stop
        if pause_event.is_set(): # If AI is paused, unpause it so it can see stop_event
            pause_event.clear()
        self.root.destroy() # Close the Tkinter window
        global gui_tk_root
        gui_tk_root = None # Indicate GUI is no longer active

    def _toggle_pause_button_action(self):
        if pause_event.is_set(): # Currently paused, so resume
            pause_event.clear()
            logging.info("AI Resumed via GUI button.")
        else: # Currently running, so pause
            pause_event.set()
            logging.info("AI Paused via GUI button.")
        # Update shared data for other parts of the app (like keyboard listener)
        with gui_data_lock:
            gui_shared_data["is_paused"] = pause_event.is_set()
            
    def _on_stop_button_press(self):
        logging.info("STOP AI button pressed from GUI.")
        stop_event.set()
        if pause_event.is_set(): # If paused, unpause to allow clean exit
            pause_event.clear()
        self.status_label.config(text="Stop signal sent. AI will halt after current operation.")
        self.stop_button.config(state=tk.DISABLED) # Disable button after click
        self.pause_button.config(state=tk.DISABLED)

    def _draw_detections_on_pil_image(self, pil_image_obj, scaled_detections_list, color="red", line_thickness=1):
        if not scaled_detections_list:
            return pil_image_obj
        
        from PIL import ImageDraw # Import here to keep it local if PIL is optional elsewhere
        # Operate on a copy if the original PIL image might be used elsewhere without drawings
        # img_with_drawings = pil_image_obj.copy() 
        # For direct modification:
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
        if not self.root or not self.root.winfo_exists(): # Check if root window still exists
            return

        with gui_data_lock: # Get a copy of shared data thread-safely
            current_data = gui_shared_data.copy()
            ai_processed_pil_image = current_data["ai_view"]
            raw_capture_pil_image = current_data["raw_capture_view"]
            current_detected_spikes = current_data.get("detected_spikes", []) # Get spikes, default to empty list

        # Update AI Processed View
        if ai_processed_pil_image:
            original_w, original_h = ai_processed_pil_image.width, ai_processed_pil_image.height
            display_w = int(original_w * config.GUI_AI_VIEW_DISPLAY_SCALE)
            display_h = int(original_h * config.GUI_AI_VIEW_DISPLAY_SCALE)
            
            # Resize the image for display
            displayable_ai_image = ai_processed_pil_image.resize((display_w, display_h), Image.Resampling.NEAREST) # NEAREST for pixelated look

            if config.GUI_MARK_DETECTED_OBJECTS and current_detected_spikes:
                # Scale spike coordinates from AI view's original size to its display size
                scaled_spike_coords = []
                for x, y, w, h in current_detected_spikes:
                    scaled_spike_coords.append((
                        int(x * config.GUI_AI_VIEW_DISPLAY_SCALE), int(y * config.GUI_AI_VIEW_DISPLAY_SCALE),
                        int(w * config.GUI_AI_VIEW_DISPLAY_SCALE), int(h * config.GUI_AI_VIEW_DISPLAY_SCALE)
                    ))
                # Draw on the *already resized* displayable image
                displayable_ai_image = self._draw_detections_on_pil_image(displayable_ai_image, scaled_spike_coords, color="orange", line_thickness=2)

            self.ai_view_photo_tk = ImageTk.PhotoImage(image=displayable_ai_image)
            self.ai_view_label.configure(image=self.ai_view_photo_tk)
            self.ai_view_label.image = self.ai_view_photo_tk # Keep reference

        # Update Raw Game Capture View
        if config.GUI_SHOW_RAW_CAPTURE and raw_capture_pil_image:
            original_w, original_h = raw_capture_pil_image.width, raw_capture_pil_image.height
            display_w = int(original_w * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            display_h = int(original_h * config.GUI_RAW_CAPTURE_DISPLAY_SCALE)
            
            displayable_raw_image = raw_capture_pil_image.resize((display_w, display_h), Image.Resampling.LANCZOS) # LANCZOS for smoother raw view
            self.raw_view_photo_tk = ImageTk.PhotoImage(image=displayable_raw_image)
            self.raw_view_label.configure(image=self.raw_view_photo_tk)
            self.raw_view_label.image = self.raw_view_photo_tk # Keep reference
            
        # Update Text Labels
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
        # Highlight Jump action if Q-value for Jump is higher
        action_display_color = "forest green" if current_data['action'] == 1 and len(current_data['q_values']) > 1 and current_data['q_values'][1] > current_data['q_values'][0] else "black"
        self.info_labels["Action"].config(text=f"Action: {action_text} ({current_data['action']})", foreground=action_display_color)
        
        self.info_labels["Loss"].config(text=f"Loss: {current_data['current_loss']:.4f}" if current_data['current_loss'] != 0 else "Loss: N/A")
        avg_loss_val = sum(current_data['avg_loss_history']) / len(current_data['avg_loss_history']) if current_data['avg_loss_history'] else 'N/A'
        self.info_labels["Avg Loss (100)"].config(text=f"Avg Loss (100): {avg_loss_val if isinstance(avg_loss_val, str) else f'{avg_loss_val:.4f}'}")
        
        self.info_labels["FPS"].config(text=current_data.get("fps_info", "AI: 0 | GUI: 0"))
        self.info_labels["Game Region"].config(text=current_data["game_region_info"])
        self.status_label.config(text=current_data["status_text"])
        
        # Update Pause/Resume button text
        self.pause_button.config(text=f"RESUME ({config.PAUSE_RESUME_KEY.upper()})" if current_data["is_paused"] else f"PAUSE ({config.PAUSE_RESUME_KEY.upper()})")

        # Calculate and update GUI FPS
        self.gui_frame_counter += 1
        current_timestamp = time.perf_counter()
        if current_timestamp - self.last_gui_update_timestamp >= 1.0: # Update FPS display every second
            actual_gui_fps = self.gui_frame_counter / (current_timestamp - self.last_gui_update_timestamp)
            self.last_gui_update_timestamp = current_timestamp
            self.gui_frame_counter = 0
            # Update shared_data with new GUI FPS, keeping AI FPS part
            with gui_data_lock:
                 current_ai_fps_part = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[0].strip()
                 gui_shared_data["fps_info"] = f"{current_ai_fps_part} | GUI: {actual_gui_fps:.1f}"
        
        # Schedule next update
        if self.root and self.root.winfo_exists(): # Check again before scheduling
            self.root.after(config.GUI_UPDATE_INTERVAL_MS, self.update_gui_elements)

def run_gui_in_thread():
    global gui_tk_root
    gui_tk_root = tk.Tk()
    app_gui_instance = AppGUI(gui_tk_root)
    gui_tk_root.mainloop() # Blocks until window is closed
    logging.info("GUI thread has finished execution.")
    # If GUI is closed by user, signal the AI loop to stop
    if not stop_event.is_set(): # Check if stop wasn't already signaled
        stop_event.set()

# --- Keyboard Listener for Pause/Resume ---
def on_key_press_event(key):
    try:
        char_pressed = key.char
    except AttributeError: # Special key (like Shift, Ctrl, etc.)
        return

    if char_pressed == config.PAUSE_RESUME_KEY:
        if pause_event.is_set(): # Currently paused
            pause_event.clear() # Resume
            logging.info("AI Resumed via keyboard press.")
        else: # Currently running
            pause_event.set() # Pause
            logging.info("AI Paused via keyboard press.")
        # Update shared data so GUI can reflect keyboard-triggered pause state
        with gui_data_lock:
            gui_shared_data["is_paused"] = pause_event.is_set()


# --- Plotting Utility ---
def plot_training_data(episode_rewards_list, episode_losses_list, episode_durations_list, save_file_path=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd # For rolling mean calculation
        plt.style.use('seaborn-v0_8-darkgrid') # Using a style that is usually available

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True) # 3 plots, shared X-axis
        
        # Rewards Plot
        rewards_series = pd.Series(episode_rewards_list)
        axes[0].plot(rewards_series, label='Episode Reward', alpha=0.6)
        axes[0].plot(rewards_series.rolling(window=100, min_periods=1).mean(), label='Avg Reward (100 episodes)', color='red', linestyle='--')
        axes[0].set_ylabel('Total Reward')
        axes[0].legend()
        axes[0].set_title('Episode Rewards Over Time')

        # Losses Plot (filter out None values if optimization didn't run)
        valid_losses = [loss_val for loss_val in episode_losses_list if loss_val is not None]
        if valid_losses:
            losses_series = pd.Series(valid_losses)
            axes[1].plot(losses_series, label='Average Loss per Episode', alpha=0.6)
            axes[1].plot(losses_series.rolling(window=100, min_periods=1).mean(), label='Avg Loss (100 episodes)', color='green', linestyle='--')
        axes[1].set_ylabel('Average Loss')
        axes[1].legend()
        axes[1].set_title('Training Loss Over Time')
        
        # Durations Plot
        durations_series = pd.Series(episode_durations_list)
        axes[2].plot(durations_series, label='Episode Duration (Steps)', alpha=0.6)
        axes[2].plot(durations_series.rolling(window=100, min_periods=1).mean(), label='Avg Duration (100 episodes)', color='purple', linestyle='--')
        axes[2].set_xlabel('Episode Number')
        axes[2].set_ylabel('Number of Steps')
        axes[2].legend()
        axes[2].set_title('Episode Durations Over Time')

        fig.suptitle(f'{config.PROJECT_NAME} Training Progress', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        
        if save_file_path:
            plt.savefig(save_file_path)
            logging.info(f"Training plots saved to {save_file_path}")
        plt.close(fig) # Close the figure to free up memory
    except ImportError:
        logging.warning("Matplotlib or Pandas not found. Skipping plot generation. Please install them to see plots.")
    except Exception as e:
        logging.error(f"An error occurred during plot generation: {e}")

# --- Main Training Loop ---
def ai_training_main_loop():
    global gui_shared_data # Allow updating shared data for GUI
    global game_region_display_window_handle # To destroy it on exit

    loop_start_timestamp = time.perf_counter()
    ai_frames_processed_this_second = 0
    last_ai_fps_update_timestamp = loop_start_timestamp

    if config.ENABLE_GUI:
        with gui_data_lock:
            gui_shared_data["status_text"] = "Initializing Environment & AI Agent..."
    
    game_env = GameEnvironment() # Initializes game window detection, templates etc.
    # Critical check: if game_env.monitor_region is still None or invalid after init, AI cannot run.
    if game_env.monitor_region is None or game_env.monitor_region.get('width', 0) == 0:
        logging.critical("FATAL ERROR: Game region could not be determined or is invalid. AI training cannot start.")
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["status_text"] = "ERROR: Game region undefined or invalid. Check logs."
        stop_event.set() # Signal all threads to stop
        return

    ai_agent = Agent(num_actions=config.NUM_ACTIONS)
    ai_agent.load_model() # Attempt to load a pre-trained model

    # Lists to store metrics for plotting
    all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations = [], [], []
    
    # Calculate delay for AI FPS limiting
    ai_loop_target_delay = 1.0 / config.AI_FPS_LIMIT if config.AI_FPS_LIMIT > 0 else 0
    
    try:
        for episode_num in range(1, config.NUM_EPISODES + 1):
            if stop_event.is_set(): # Check if stop was signaled
                logging.info("Stop event received in AI training loop. Terminating.")
                break
            
            if pause_event.is_set(): # Check if paused
                logging.info("AI Training is Paused...")
                current_status_text = f"AI Training Paused (Press '{config.PAUSE_RESUME_KEY.upper()}' to resume)"
                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["status_text"] = current_status_text
                pause_event.wait() # Block here until pause_event.clear() is called
                logging.info("AI Training Resumed.")
                if config.ENABLE_GUI: # Ensure GUI reflects resumed state
                    with gui_data_lock:
                        gui_shared_data["is_paused"] = False 
                        gui_shared_data["status_text"] = f"Resuming Ep. {episode_num}..."


            current_state_tensor, _ = game_env.reset() # Reset environment for new episode
            current_episode_total_reward = 0.0
            current_episode_loss_sum = 0.0
            current_episode_optimization_steps = 0
            
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["episode"] = episode_num
                    gui_shared_data["current_episode_reward"] = 0.0 # Reset for GUI display
                    gui_shared_data["status_text"] = f"Running Episode {episode_num}..."

            for step_num_in_episode in range(1, config.MAX_STEPS_PER_EPISODE + 1):
                if stop_event.is_set(): break # Exit step loop if stop signaled
                if pause_event.is_set(): # Handle mid-episode pause
                    mid_ep_pause_status = f"AI Paused (Ep. {episode_num}, Step {step_num_in_episode})"
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["status_text"] = mid_ep_pause_status
                    pause_event.wait()
                    if config.ENABLE_GUI:
                        with gui_data_lock: gui_shared_data["is_paused"] = False; gui_shared_data["status_text"] = f"Resuming Ep. {episode_num}..."

                
                step_processing_start_time = time.perf_counter()
                
                action_tensor = ai_agent.select_action(current_state_tensor) # AI selects action
                next_state_tensor, reward_value, done_flag, _ = game_env.step(action_tensor.item()) # Perform action in env
                
                # Apply progress reward if configured and not done
                if config.REWARD_PROGRESS_FACTOR != 0 and not done_flag:
                    progress_reward = config.REWARD_PROGRESS_FACTOR * (step_num_in_episode / config.MAX_STEPS_PER_EPISODE)
                    reward_value += progress_reward

                current_episode_total_reward += reward_value
                
                # Store transition in replay memory
                ai_agent.memory.push(current_state_tensor, action_tensor,
                                  None if done_flag else next_state_tensor, # next_state is None if terminal
                                  torch.tensor([reward_value], device=config.DEVICE, dtype=torch.float),
                                  torch.tensor([done_flag], device=config.DEVICE, dtype=torch.bool) 
                                  )
                current_state_tensor = next_state_tensor # Move to next state
                
                # Perform one step of optimization on the policy network
                loss_item_value = ai_agent.optimize_model()
                if loss_item_value is not None:
                    current_episode_loss_sum += loss_item_value
                    current_episode_optimization_steps += 1

                # Update GUI with current step info
                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["step"] = step_num_in_episode
                        gui_shared_data["total_steps"] = ai_agent.total_steps_done_in_training
                        gui_shared_data["current_episode_reward"] = current_episode_total_reward

                # AI FPS Limiter
                if ai_loop_target_delay > 0:
                    time_elapsed_this_step = time.perf_counter() - step_processing_start_time
                    sleep_duration_needed = ai_loop_target_delay - time_elapsed_this_step
                    if sleep_duration_needed > 0:
                        time.sleep(sleep_duration_needed)
                
                # AI FPS Calculation for GUI
                ai_frames_processed_this_second +=1
                current_loop_timestamp = time.perf_counter()
                if current_loop_timestamp - last_ai_fps_update_timestamp >= 1.0: # Update AI FPS display every second
                    actual_ai_fps_val = ai_frames_processed_this_second / (current_loop_timestamp - last_ai_fps_update_timestamp)
                    last_ai_fps_update_timestamp = current_loop_timestamp
                    ai_frames_processed_this_second = 0
                    if config.ENABLE_GUI: # Update shared data for GUI to pick up
                        with gui_data_lock:
                            current_gui_fps_part = gui_shared_data.get("fps_info", "AI: 0 | GUI: 0").split("|")[-1].strip() # Keep GUI part
                            gui_shared_data["fps_info"] = f"AI: {actual_ai_fps_val:.1f} | {current_gui_fps_part}"

                if done_flag: # End episode if 'done'
                    break 
            
            # --- End of Episode Logic ---
            if stop_event.is_set(): break # Exit episode loop if stop signaled

            all_episode_total_rewards.append(current_episode_total_reward)
            all_episode_step_durations.append(step_num_in_episode) # Actual steps taken
            avg_loss_this_episode = current_episode_loss_sum / current_episode_optimization_steps if current_episode_optimization_steps > 0 else None
            all_episode_avg_losses.append(avg_loss_this_episode)

            # Update GUI history deques
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["avg_reward_history"].append(current_episode_total_reward)
                    if avg_loss_this_episode is not None:
                        gui_shared_data["avg_loss_history"].append(avg_loss_this_episode)
                    gui_shared_data["status_text"] = f"Episode {episode_num} Finished. Reward: {current_episode_total_reward:.2f}"

            logging.info(f"Episode {episode_num} Summary: Steps={step_num_in_episode}, Reward={current_episode_total_reward:.2f}, AvgLoss={avg_loss_this_episode if avg_loss_this_episode is not None else 'N/A'}, Epsilon={gui_shared_data['epsilon']:.4f}, MemorySize={len(ai_agent.memory)}")
            
            # Update target network periodically
            if episode_num % config.TARGET_UPDATE_FREQ_EPISODES == 0:
                ai_agent.update_target_net()
                logging.info("Target network updated.")
            
            # Save model and plots periodically
            if episode_num % config.SAVE_MODEL_EVERY_N_EPISODES == 0:
                 ai_agent.save_model()
                 if all_episode_total_rewards: # Only plot if there's data
                    plot_training_data(all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations)
        
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt (Ctrl+C) detected in AI training loop. Initiating stop.")
    except Exception as e:
        logging.error(f"UNEXPECTED ERROR IN AI TRAINING LOOP: {e}", exc_info=True) # Log full traceback
    finally:
        stop_event.set() # Ensure stop_event is set in all exit paths
        logging.info("AI Training Loop finalizing procedures...")
        ai_agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth") # Save final model
        if all_episode_total_rewards: # Plot final results if data exists
            plot_training_data(all_episode_total_rewards, all_episode_avg_losses, all_episode_step_durations, 
                               f"{config.PROJECT_NAME.lower()}_final_training_plots.png")
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["status_text"] = "AI Training Loop Finished. GUI can be closed."
        logging.info("AI Training Loop has finished.")
        
        # Clean up the win32gui overlay window if it exists
        if game_region_display_window_handle and win32gui:
            try:
                win32gui.DestroyWindow(game_region_display_window_handle)
            except Exception: pass # Suppress error if already destroyed
            game_region_display_window_handle = None


if __name__ == '__main__':
    logging.info(f"--- {config.PROJECT_NAME} AI Main Script Starting --- PID: {os.getpid()}")
    
    # Start keyboard listener for pause/resume
    keyboard_listener_thread = pynput_keyboard.Listener(on_press=on_key_press_event)
    keyboard_listener_thread.start()
    logging.info(f"Keyboard listener started. Press '{config.PAUSE_RESUME_KEY}' to Pause/Resume AI.")

    # Countdown before starting, giving user time to focus game window
    for i in range(2, 0, -1): # Shortened countdown
        logging.info(f"Please focus the Geometry Dash game window... Starting AI in {i} seconds.")
        time.sleep(1)

    # Start GUI thread if enabled
    gui_main_thread = None
    if config.ENABLE_GUI:
        gui_main_thread = threading.Thread(target=run_gui_in_thread, daemon=True) # Daemon so it exits with main
        gui_main_thread.start()
        logging.info("GUI thread has been initiated.")
        time.sleep(0.5) # Brief pause to allow GUI to initialize before AI starts heavy work

    # Start AI training thread
    ai_main_thread = threading.Thread(target=ai_training_main_loop, daemon=True) # Daemon
    ai_main_thread.start()
    logging.info("AI training thread has been initiated.")

    # Keep the main thread alive to manage child threads and handle graceful shutdown
    try:
        while ai_main_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.5) # Check periodically
            # If GUI was enabled but its window closed (gui_tk_root becomes None), signal stop
            if config.ENABLE_GUI and gui_tk_root is None and gui_main_thread and gui_main_thread.is_alive():
                logging.info("GUI window appears to have been closed. Signaling AI thread to stop.")
                stop_event.set() # This will break the AI loop
                break 
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected in main thread. Initiating graceful shutdown.")
        stop_event.set() # Signal all threads to stop
    
    logging.info("Main thread: Waiting for AI training thread to complete...")
    if pause_event.is_set(): # If paused, clear event so AI thread can exit its wait
        pause_event.clear()
    ai_main_thread.join(timeout=10) # Wait for AI thread with a timeout
    if ai_main_thread.is_alive():
        logging.warning("AI training thread did not stop gracefully within the timeout. It might be stuck.")

    if config.ENABLE_GUI and gui_main_thread and gui_main_thread.is_alive():
        logging.info("Main thread: Waiting for GUI thread to complete...")
        # If GUI root still exists, try to destroy it to close the window
        if gui_tk_root and gui_tk_root.winfo_exists():
             gui_tk_root.destroy()
        gui_main_thread.join(timeout=5) # Wait for GUI thread with a timeout
    
    keyboard_listener_thread.stop() # Stop the keyboard listener
    logging.info("Keyboard listener has been stopped.")
    logging.info(f"--- {config.PROJECT_NAME} AI Shutdown Sequence Complete ---")

#Вызов духа здравого смысла. Он просит тебя: "Комментируй, сука"
