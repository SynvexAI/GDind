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
    logging.warning("pygetwindow not installed. Window auto-detection will be disabled. "
                    "Install with 'pip install pygetwindow'")
try:
    from pywinauto.application import Application 
    
except ImportError:
    Application = None
    logging.warning("pywinauto not installed. Advanced window focusing might be limited. "
                    "Install with 'pip install pywinauto' (Windows only)")


import config 



def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=level, format=log_format, handlers=[
        logging.FileHandler(config.LOG_FILE, mode='w'),
        logging.StreamHandler()
    ])
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('pygetwindow').setLevel(logging.WARNING) 
    logging.info(f"Logging setup complete. Log level: {config.LOG_LEVEL.upper()}")
    logging.info(f"Using device: {config.DEVICE}")

setup_logging()



gui_tk_root = None
gui_data_lock = threading.Lock() 
gui_shared_data = {
    "ai_view": None, 
    "q_values": [0.0, 0.0],
    "action": 0,
    "epsilon": 0.0,
    "episode": 0,
    "step": 0,
    "reward": 0.0,
    "status_text": "Initializing..."
}
stop_event = threading.Event() 



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        logging.info(f"ReplayMemory initialized with capacity: {capacity}")
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)



class DQN(nn.Module):
    def __init__(self, h, w, outputs, num_frames=1):
        super(DQN, self).__init__()
        in_channels = (1 if config.GRAYSCALE else 3) * num_frames
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        def conv2d_size_out(size, kernel_size, stride): return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        logging.info(f"DQN model initialized. Input: ({in_channels}, {h}, {w}), Output: {outputs}")
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
            logging.error("Geometry Dash window not found or could not be focused. "
                          f"Using fallback region: {config.FALLBACK_GAME_REGION} if set, or first monitor.")
            self.monitor_region = config.FALLBACK_GAME_REGION if config.FALLBACK_GAME_REGION else self.sct.monitors[1]
        
        logging.info(f"GameEnvironment initialized. Screen region: {self.monitor_region}")

        
        self.game_over_template = self._load_template(config.GAME_OVER_TEMPLATE_PATH, "Game Over")
        
        self.player_icon_template = self._load_template(config.PLAYER_ICON_TEMPLATE_PATH, "Player Icon")
        if self.player_icon_template is not None and config.PLAYER_EXPECTED_REGION is None:
            logging.warning("Player icon template loaded, but PLAYER_EXPECTED_REGION is not set in config. Player detection might be unreliable.")


        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)

        if config.DEBUG_VISUALIZE_AI_INPUT_SEPARATELY and not config.ENABLE_GUI:
            cv2.namedWindow("AI Input CV2", cv2.WINDOW_NORMAL) 

    def _load_template(self, path, template_name):
        if not path or not os.path.exists(path):
            logging.warning(f"{template_name} template not found or path not specified: {path}. "
                            f"Related detection will be disabled.")
            return None
        template = cv2.imread(path)
        if template is None:
            logging.error(f"Failed to load {template_name} template from {path}")
            return None
        if config.GRAYSCALE:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        logging.info(f"Loaded {template_name} template from: {path}, shape: {template.shape}")
        return template

    def _get_and_focus_game_window(self):
        if not pygetwindow:
            logging.warning("pygetwindow not available, cannot auto-detect window.")
            return None

        try:
            gd_windows = pygetwindow.getWindowsWithTitle(config.WINDOW_TITLE_SUBSTRING)
            if not gd_windows:
                logging.warning(f"No window with title containing '{config.WINDOW_TITLE_SUBSTRING}' found.")
                return None
            
            gd_window = gd_windows[0] 
            logging.info(f"Found window: {gd_window.title}")

            if gd_window.isMinimized:
                gd_window.restore()
            

            
            if Application and os.name == 'nt':
                try:
                    
                    app = Application().connect(handle=gd_window._hWnd)
                    app_window = app.window(handle=gd_window._hWnd)
                    if app_window.exists():
                        app_window.set_focus()
                        logging.info(f"Focused window '{gd_window.title}' using pywinauto.")
                    else:
                        logging.warning("pywinauto could not find window by handle after pygetwindow found it.")
                        gd_window.activate() 
                except Exception as e:
                    logging.warning(f"pywinauto error focusing window: {e}. Falling back to pygetwindow activate().")
                    gd_window.activate()
            else:
                 gd_window.activate() 

            time.sleep(0.5) 

            
            
            return {"top": gd_window.top, "left": gd_window.left, 
                    "width": gd_window.width, "height": gd_window.height,
                    "monitor": 1} 

        except Exception as e:
            logging.error(f"Error auto-detecting/focusing Geometry Dash window: {e}")
            return None


    def _capture_frame(self):
        
        try:
            sct_img = self.sct.grab(self.monitor_region)
            img = np.array(sct_img)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except mss.exception.ScreenShotError as e:
            logging.error(f"Screen capture error: {e}. Retrying...")
            time.sleep(0.1)
            return self._capture_frame()

    def _preprocess_frame(self, frame_bgr):
        
        if config.GRAYSCALE: frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else: frame = frame_bgr
        frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        return frame_resized.astype(np.uint8) 

    def _stack_frames(self, processed_frame_hw_or_hwc): 
        
        if config.GRAYSCALE:
            
            frame_for_stack = np.expand_dims(processed_frame_hw_or_hwc, axis=0)
        else:
            
            frame_for_stack = np.transpose(processed_frame_hw_or_hwc, (2, 0, 1))

        if not self.stacked_frames:
            for _ in range(config.NUM_FRAMES_STACKED):
                self.stacked_frames.append(frame_for_stack)
        else:
            self.stacked_frames.append(frame_for_stack)
        
        
        
        
        stacked_state_chw = np.concatenate(list(self.stacked_frames), axis=0)
        return torch.from_numpy(stacked_state_chw).unsqueeze(0).to(config.DEVICE).float() 


    def reset(self):
        
        logging.debug("Resetting environment.")
        self.stacked_frames.clear()
        raw_frame = self._capture_frame()
        processed_frame = self._preprocess_frame(raw_frame) 
        
        
        
        
        if config.GRAYSCALE: frame_for_deque = np.expand_dims(processed_frame, axis=0)
        else: frame_for_deque = np.transpose(processed_frame, (2,0,1))
        
        for _ in range(config.NUM_FRAMES_STACKED):
             self.stacked_frames.append(frame_for_deque) 

        
        stacked_state_tensor = torch.from_numpy(np.concatenate(list(self.stacked_frames), axis=0)).unsqueeze(0).to(config.DEVICE).float()
        
        return stacked_state_tensor, raw_frame


    def step(self, action_tensor):
        action = action_tensor.item()
        if action == 1:
            self.mouse.press(Button.left)
            time.sleep(config.JUMP_DURATION)
            self.mouse.release(Button.left)
        if config.ACTION_DELAY > 0: time.sleep(config.ACTION_DELAY)

        raw_next_frame = self._capture_frame()
        processed_next_frame = self._preprocess_frame(raw_next_frame) 
        next_state_tensor = self._stack_frames(processed_next_frame) 

        reward, done = self._get_reward_and_done(raw_next_frame) 

        
        if config.ENABLE_GUI:
            with gui_data_lock:
                
                
                if config.GRAYSCALE:
                    
                    gui_shared_data["ai_view"] = Image.fromarray(processed_next_frame, 'L')
                else:
                    
                    gui_shared_data["ai_view"] = Image.fromarray(cv2.cvtColor(processed_next_frame, cv2.COLOR_BGR2RGB))
                

        
        if config.DEBUG_VISUALIZE_AI_INPUT_SEPARATELY and not config.ENABLE_GUI:
            display_frame_cv2 = processed_next_frame
            if not config.GRAYSCALE: 
                 pass 
            
            scale = config.VISUALIZATION_WINDOW_SCALE
            display_frame_cv2_resized = cv2.resize(display_frame_cv2, (config.FRAME_WIDTH*scale, config.FRAME_HEIGHT*scale), cv2.INTER_NEAREST)
            cv2.imshow("AI Input CV2", display_frame_cv2_resized)
            cv2.waitKey(1)
            
        return next_state_tensor, reward, done, raw_next_frame


    def _match_template(self, frame_area, template, threshold):
        if template is None or frame_area.shape[0] < template.shape[0] or frame_area.shape[1] < template.shape[1]:
            return False, 0.0
        
        res = cv2.matchTemplate(frame_area, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val > threshold, max_val

    def _get_reward_and_done(self, current_raw_frame):
        done = False
        reward = config.REWARD_ALIVE

        frame_for_detection = current_raw_frame
        if config.GRAYSCALE:
            frame_for_detection_gray = cv2.cvtColor(current_raw_frame, cv2.COLOR_BGR2GRAY)
        else:
            
            frame_for_detection_gray = None 

        
        if self.game_over_template is not None:
            source_img_for_go = frame_for_detection_gray if config.GRAYSCALE else frame_for_detection
            
            search_area_go = source_img_for_go
            if config.GAME_OVER_SEARCH_REGION:
                x, y, w, h = config.GAME_OVER_SEARCH_REGION
                search_area_go = source_img_for_go[y:y+h, x:x+w]

            is_game_over, match_val_go = self._match_template(search_area_go, self.game_over_template, config.GAME_OVER_DETECTION_THRESHOLD)
            if is_game_over:
                logging.info(f"Game Over detected (match: {match_val_go:.2f}).")
                done = True
                reward = config.REWARD_DEATH
                if config.SAVE_FRAMES_ON_DEATH:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(f"death_frame_go_{ts}.png", current_raw_frame)
                return reward, done 

        
        
        
        if not done and self.player_icon_template is not None and config.PLAYER_EXPECTED_REGION:
            source_img_for_player = frame_for_detection_gray if config.GRAYSCALE else frame_for_detection
            
            px, py, pw, ph = config.PLAYER_EXPECTED_REGION
            
            max_h, max_w = source_img_for_player.shape[:2]
            px, py = max(0, px), max(0, py)
            pw, ph = min(pw, max_w - px), min(ph, max_h - py)

            if pw > 0 and ph > 0 : 
                player_search_area = source_img_for_player[py:py+ph, px:px+pw]
                player_found, match_val_player = self._match_template(player_search_area, self.player_icon_template, config.PLAYER_ICON_DETECTION_THRESHOLD)

                if not player_found:
                    
                    
                    logging.debug(f"Player icon NOT found in expected region (match: {match_val_player:.2f}). Applying penalty.")
                    reward += config.PENALTY_PLAYER_NOT_FOUND 
                    
                    
                    if config.SAVE_FRAMES_ON_DEATH: 
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(f"player_not_found_{ts}.png", current_raw_frame)
                else:
                    logging.debug(f"Player icon found (match: {match_val_player:.2f}).")
            else:
                logging.warning("PLAYER_EXPECTED_REGION is invalid or outside the frame.")

        return reward, done


class Agent:
    
    def __init__(self, num_actions, env_for_shape_calc):
        self.num_actions = num_actions
        initial_state, _ = env_for_shape_calc.reset()
        _, c_total, h, w = initial_state.shape
        self.policy_net = DQN(h, w, num_actions, num_frames=1).to(config.DEVICE) 
        
        
        
        
        
        
        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, num_frames=config.NUM_FRAMES_STACKED).to(config.DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, num_frames=config.NUM_FRAMES_STACKED).to(config.DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.total_steps_done = 0
        logging.info("Agent initialized.")

    def select_action(self, state_tensor): 
        self.total_steps_done += 1
        current_epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                       np.exp(-1. * self.total_steps_done / config.EPSILON_DECAY_FRAMES)
        
        action_tensor_val = None
        q_values_for_gui = [0.0, 0.0] 

        if random.random() > current_epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor) 
                action_tensor_val = q_values.max(1)[1].view(1, 1)
                q_values_for_gui = q_values.cpu().squeeze().tolist()
        else:
            action_tensor_val = torch.tensor([[random.randrange(self.num_actions)]], device=config.DEVICE, dtype=torch.long)
            
        
        
        if config.ENABLE_GUI:
            with gui_data_lock:
                gui_shared_data["q_values"] = q_values_for_gui
                gui_shared_data["action"] = action_tensor_val.item()
                gui_shared_data["epsilon"] = current_epsilon
        
        return action_tensor_val

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or self.total_steps_done < config.LEARN_START_STEPS:
            return None
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.DEVICE, dtype=torch.bool)
        
        safe_non_final_next_states = [s for s in batch.next_state if s is not None]
        if not safe_non_final_next_states: 
            non_final_next_states_cat = None
        else:
            non_final_next_states_cat = torch.cat(safe_non_final_next_states)

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
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    
    def save_model(self, path=config.MODEL_SAVE_PATH):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps_done': self.total_steps_done,
            }, path)
            logging.info(f"Model saved to {path}")
        except Exception as e: logging.error(f"Error saving model: {e}")
    def load_model(self, path=config.MODEL_SAVE_PATH):
        if not os.path.exists(path):
            logging.warning(f"Model file not found at {path}. Starting with a new model.")
            return False
        try:
            checkpoint = torch.load(path, map_location=config.DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps_done = checkpoint.get('total_steps_done', 0)
            self.policy_net.to(config.DEVICE); self.target_net.to(config.DEVICE); self.target_net.eval()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(config.DEVICE)
            logging.info(f"Model loaded from {path}. Resuming from {self.total_steps_done} total steps.")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}. Starting with a new model.")
            return False


class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{config.PROJECT_NAME} Dashboard")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing) 

        
        style = ttk.Style()
        style.theme_use('clam') 

        
        self.view_frame = ttk.LabelFrame(root, text="AI Vision")
        self.view_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.ai_view_label = ttk.Label(self.view_frame)
        self.ai_view_label.pack(padx=5, pady=5)
        
        placeholder_img = Image.new('RGB', (config.FRAME_WIDTH * 2, config.FRAME_HEIGHT * 2), color = 'grey')
        self.ai_view_photo = ImageTk.PhotoImage(image=placeholder_img)
        self.ai_view_label.configure(image=self.ai_view_photo)


        
        self.info_frame = ttk.LabelFrame(root, text="AI Stats")
        self.info_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.q_label = ttk.Label(self.info_frame, text="Q-Values: [0.00, 0.00]")
        self.q_label.pack(anchor="w", padx=5)
        self.action_label = ttk.Label(self.info_frame, text="Action: N/A")
        self.action_label.pack(anchor="w", padx=5)
        self.epsilon_label = ttk.Label(self.info_frame, text="Epsilon: 1.0000")
        self.epsilon_label.pack(anchor="w", padx=5)
        self.episode_label = ttk.Label(self.info_frame, text="Episode: 0 / 0")
        self.episode_label.pack(anchor="w", padx=5)
        self.step_label = ttk.Label(self.info_frame, text="Step: 0")
        self.step_label.pack(anchor="w", padx=5)
        self.reward_label = ttk.Label(self.info_frame, text="Episode Reward: 0.0")
        self.reward_label.pack(anchor="w", padx=5)
        
        self.status_label_title = ttk.Label(self.info_frame, text="Status:")
        self.status_label_title.pack(anchor="w", padx=5, pady=(10,0))
        self.status_label = ttk.Label(self.info_frame, text="Initializing...", wraplength=200)
        self.status_label.pack(anchor="w", padx=5)

        
        self.stop_button = ttk.Button(root, text="STOP AI", command=self._on_stop_button)
        self.stop_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.update_gui()

    def _on_closing(self):
        logging.info("GUI window closed by user.")
        stop_event.set() 
        self.root.destroy()

    def _on_stop_button(self):
        logging.info("STOP AI button pressed.")
        stop_event.set()
        self.status_label.config(text="Stop signal sent. AI will halt after current step/episode.")
        self.stop_button.config(state=tk.DISABLED)


    def update_gui(self):
        if not self.root.winfo_exists(): 
            return

        with gui_data_lock:
            data = gui_shared_data.copy() 

        if data["ai_view"]:
            
            w, h = data["ai_view"].width, data["ai_view"].height
            display_w, display_h = w * config.VISUALIZATION_WINDOW_SCALE, h * config.VISUALIZATION_WINDOW_SCALE
            
            
            img_resized = data["ai_view"].resize((display_w, display_h), Image.Resampling.LANCZOS)
            self.ai_view_photo = ImageTk.PhotoImage(image=img_resized)
            self.ai_view_label.configure(image=self.ai_view_photo)

        q_text = f"Q-Values: [{data['q_values'][0]:.2f} (Do Nothing), {data['q_values'][1]:.2f} (Jump)]"
        self.q_label.config(text=q_text)
        action_text = "JUMP" if data['action'] == 1 else "DO NOTHING"
        self.action_label.config(text=f"Action: {action_text} ({data['action']})")
        self.epsilon_label.config(text=f"Epsilon: {data['epsilon']:.4f}")
        self.episode_label.config(text=f"Episode: {data['episode']} / {config.NUM_EPISODES}")
        self.step_label.config(text=f"Step: {data['step']}")
        self.reward_label.config(text=f"Ep. Reward: {data['reward']:.2f}")
        self.status_label.config(text=data['status_text'])

        self.root.after(config.GUI_UPDATE_INTERVAL_MS, self.update_gui)


def run_gui():
    global gui_tk_root
    gui_tk_root = tk.Tk()
    app = AppGUI(gui_tk_root)
    gui_tk_root.mainloop()
    logging.info("GUI thread finished.")




def plot_training_results(episode_rewards, episode_losses, episode_durations, save_path=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        axs[0].plot(episode_rewards, label='Total Reward per Episode', color='dodgerblue')
        axs[0].plot(pd.Series(episode_rewards).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 eps)', color='orangered', linestyle='--')
        axs[0].set_ylabel('Total Reward'); axs[0].legend(); axs[0].set_title('Training Rewards')
        valid_losses = [l for l in episode_losses if l is not None]
        if valid_losses:
            axs[1].plot(valid_losses, label='Average Loss per Episode', color='mediumseagreen')
            axs[1].plot(pd.Series(valid_losses).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 eps)', color='purple', linestyle='--')
        axs[1].set_ylabel('Average Loss'); axs[1].legend(); axs[1].set_title('Training Loss')
        axs[2].plot(episode_durations, label='Steps per Episode', color='goldenrod')
        axs[2].plot(pd.Series(episode_durations).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 eps)', color='saddlebrown', linestyle='--')
        axs[2].set_xlabel('Episode'); axs[2].set_ylabel('Number of Steps'); axs[2].legend(); axs[2].set_title('Episode Durations')
        fig.suptitle(f'{config.PROJECT_NAME} - Training Progress', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path: plt.savefig(save_path); logging.info(f"Training plots saved to {save_path}")
    except ImportError: logging.warning("Matplotlib or pandas not found. Skipping plot generation.")
    except Exception as e: logging.error(f"Error plotting results: {e}")


def main_training_loop():
    global gui_shared_data 
    logging.info(f"Starting {config.PROJECT_NAME} AI Training Loop...")
    
    if config.ENABLE_GUI:
        with gui_data_lock: gui_shared_data["status_text"] = "Initializing Environment..."
    
    env = GameEnvironment() 
    if env.monitor_region is None and config.FALLBACK_GAME_REGION is None:
        logging.error("Failed to define game region. Exiting.")
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = "ERROR: Failed to define game region."
        return

    agent = Agent(num_actions=config.NUM_ACTIONS, env_for_shape_calc=env)
    agent.load_model()

    all_episode_rewards, all_episode_avg_losses, all_episode_durations = [], [], []
    frame_limiter_delay = 1.0 / config.FPS_LIMIT if config.FPS_LIMIT > 0 else 0

    try:
        for i_episode in range(1, config.NUM_EPISODES + 1):
            if stop_event.is_set():
                logging.info("Stop event received in main loop. Breaking.")
                break
            
            if config.ENABLE_GUI:
                 with gui_data_lock: gui_shared_data["status_text"] = f"Episode {i_episode}: Resetting..."

            current_state_tensor, _ = env.reset()
            episode_reward = 0
            episode_loss_sum = 0
            episode_optimization_steps = 0
            
            current_epsilon_val = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                           np.exp(-1. * agent.total_steps_done / config.EPSILON_DECAY_FRAMES)
            logging.info(f"--- Episode {i_episode}/{config.NUM_EPISODES} --- Epsilon: {current_epsilon_val:.4f}")
            if config.ENABLE_GUI:
                with gui_data_lock:
                    gui_shared_data["episode"] = i_episode
                    gui_shared_data["reward"] = 0.0 

            for t_step in range(config.MAX_STEPS_PER_EPISODE):
                if stop_event.is_set(): break
                step_start_time = time.perf_counter()
                
                if config.ENABLE_GUI:
                    with gui_data_lock:
                        gui_shared_data["step"] = t_step + 1
                        gui_shared_data["status_text"] = f"Episode {i_episode}, Step {t_step+1}: Running..."
                
                action_tensor = agent.select_action(current_state_tensor) 
                next_state_tensor, reward_val, done, _ = env.step(action_tensor) 

                episode_reward += reward_val
                if config.ENABLE_GUI:
                    with gui_data_lock: gui_shared_data["reward"] = episode_reward
                
                reward_tensor = torch.tensor([reward_val], device=config.DEVICE, dtype=torch.float)
                

                agent.memory.push(current_state_tensor, action_tensor,
                                  None if done else next_state_tensor,
                                  reward_tensor,
                                  torch.tensor([done], device=config.DEVICE, dtype=torch.bool) 
                                  )
                current_state_tensor = next_state_tensor
                
                loss = agent.optimize_model()
                if loss is not None:
                    episode_loss_sum += loss
                    episode_optimization_steps += 1
                
                if frame_limiter_delay > 0:
                    elapsed_step_time = time.perf_counter() - step_start_time
                    sleep_time = frame_limiter_delay - elapsed_step_time
                    if sleep_time > 0: time.sleep(sleep_time)

                if done: break
            
            
            if stop_event.is_set():
                logging.info("Stop event detected after episode completion step. Halting training.")
                break

            all_episode_rewards.append(episode_reward)
            all_episode_durations.append(t_step + 1)
            avg_loss = episode_loss_sum / episode_optimization_steps if episode_optimization_steps > 0 else None
            all_episode_avg_losses.append(avg_loss)

            logging.info(f"Episode {i_episode} finished. Steps: {t_step + 1}, Reward: {episode_reward:.2f}, Avg Loss: {avg_loss if avg_loss is not None else 'N/A'}")
            if config.ENABLE_GUI:
                 with gui_data_lock: gui_shared_data["status_text"] = f"Episode {i_episode} DONE. Reward: {episode_reward:.2f}"


            if i_episode % config.TARGET_UPDATE_FREQ_EPISODES == 0:
                agent.update_target_net()
            if i_episode % config.SAVE_MODEL_EVERY_N_EPISODES == 0:
                 agent.save_model()
                 plot_training_results(all_episode_rewards, all_episode_avg_losses, all_episode_durations)

    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user (Ctrl+C).")
        stop_event.set() 
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        stop_event.set()
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = f"ERROR: {e}"
    finally:
        logging.info("Performing final cleanup and saving...")
        agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth")
        if cv2.getWindowProperty("AI Input CV2", 0) >= 0 and config.DEBUG_VISUALIZE_AI_INPUT_SEPARATELY:
             cv2.destroyWindow("AI Input CV2")
        plot_training_results(all_episode_rewards, all_episode_avg_losses, all_episode_durations, f"{config.PROJECT_NAME.lower()}_final_plots.png")
        logging.info("Training loop finished.")
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = "Training finished. You can close this window."
            


if __name__ == '__main__':
    logging.info(f"Main thread started. PID: {os.getpid()}")
    
    
    for i in range(3, 0, -1):
        logging.info(f"Game starting in {i}...")
        if config.ENABLE_GUI:
            with gui_data_lock: gui_shared_data["status_text"] = f"Game starting in {i}..."
        time.sleep(1)

    if config.ENABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, daemon=True)
        gui_thread.start()
        logging.info("GUI thread started.")
        
        time.sleep(1) 

    main_training_loop_thread = threading.Thread(target=main_training_loop, daemon=True)
    main_training_loop_thread.start()
    
    
    main_training_loop_thread.join() 
    
    if config.ENABLE_GUI and gui_thread.is_alive():
        logging.info("Main training loop finished. GUI is still running. Close GUI to exit completely.")
        
        
        
        gui_thread.join() 
    
    logging.info("Program shutdown complete.")