import time
import logging
import os
import threading
import collections
import random
from typing import Any, Dict, Tuple, List, Optional, Callable

import cv2
import mss
import numpy as np
import pygetwindow as gw
from pynput.mouse import Button as MouseButton, Controller as MouseController
import keyboard
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


bot_paused = False
bot_running = True


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            if not config_data:
                raise ValueError("Config file is empty or invalid.")
            logging.info(f"Configuration loaded from {self.config_path}")
            return config_data
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration: {e}")
            raise
        except ValueError as e:
            logging.error(f"Error in configuration data: {e}")
            raise


    def get_setting(self, section: str, key: str, default: Optional[Any] = None) -> Any:
        try:
            return self.config[section][key]
        except KeyError:
            if default is not None:
                logging.warning(f"Config key '{key}' not found in section '{section}'. Using default: {default}")
                return default
            logging.error(f"Config key '{key}' not found in section '{section}' and no default provided.")
            raise

    def get_section(self, section: str) -> Dict[str, Any]:
        try:
            return self.config[section]
        except KeyError:
            logging.error(f"Config section '{section}' not found.")
            raise



def setup_logging(config: ConfigManager):
    log_config = config.get_section("logging")
    log_file = log_config.get("log_file", "gdint.log")
    console_level_str = log_config.get("console_level", "INFO").upper()
    file_level_str = log_config.get("file_level", "DEBUG").upper()

    console_level = getattr(logging, console_level_str, logging.INFO)
    file_level = getattr(logging, file_level_str, logging.DEBUG)

    logging.basicConfig(
        level=min(console_level, file_level), 
        format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler() 
        ]
    )
    
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(file_level)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)
    
    logging.info("Logging initialized.")



class ScreenCapture:
    def __init__(self, config: ConfigManager):
        self.config = config.get_section("capture")
        self.window_title = self.config.get("window_title", "Geometry Dash")
        self.target_fps = self.config.get("frame_rate", 30)
        self.frame_interval = 1.0 / self.target_fps
        self.sct = mss.mss()
        self.game_window = None
        self.monitor_info = None 
        self.mouse = MouseController()

    def find_and_focus_window(self) -> bool:
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            logging.error(f"Window with title '{self.window_title}' not found.")
            return False
        
        self.game_window = windows[0]
        try:
            if self.game_window.isMinimized:
                self.game_window.restore()
            self.game_window.activate()
            time.sleep(0.5) 
            self.monitor_info = {
                "top": self.game_window.top,
                "left": self.game_window.left,
                "width": self.game_window.width,
                "height": self.game_window.height,
            }
            logging.info(f"Found and focused window: {self.game_window.title} at {self.monitor_info}")
            self._center_cursor()
            return True
        except Exception as e: 
            logging.error(f"Error focusing window or getting dimensions: {e}")
            self.game_window = None
            self.monitor_info = None
            return False

    def _center_cursor(self):
        if self.game_window and self.monitor_info:
            center_x = self.monitor_info["left"] + self.monitor_info["width"] // 2
            center_y = self.monitor_info["top"] + self.monitor_info["height"] // 2
            self.mouse.position = (center_x, center_y)
            logging.debug(f"Cursor centered at ({center_x}, {center_y})")

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.game_window or not self.monitor_info:
            if not self.find_and_focus_window(): 
                 logging.warning("Game window not available for capture.")
                 return None
        
        try:
            
            if self.game_window.top != self.monitor_info["top"] or \
               self.game_window.left != self.monitor_info["left"] or \
               self.game_window.width != self.monitor_info["width"] or \
               self.game_window.height != self.monitor_info["height"]:
                logging.info("Game window moved or resized. Re-acquiring...")
                if not self.find_and_focus_window():
                    return None

            img = self.sct.grab(self.monitor_info)
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) 
            return frame_bgr
        except Exception as e:
            logging.error(f"Error during screen capture: {e}")
            
            self.game_window = None 
            self.monitor_info = None
            return None

    def get_window_dimensions(self) -> Optional[Tuple[int, int]]:
        if self.monitor_info:
            return self.monitor_info["width"], self.monitor_info["height"]
        return None



class DuelingCNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super(DuelingCNN, self).__init__()
        c, h, w = input_shape 

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        
        conv_out_size = self._get_conv_out_size(input_shape)

        
        self.advantage_fc1 = nn.Linear(conv_out_size, 512)
        self.advantage_fc2 = nn.Linear(512, num_actions)

        
        self.value_fc1 = nn.Linear(conv_out_size, 512)
        self.value_fc2 = nn.Linear(512, 1)

    def _get_conv_out_size(self, shape: Tuple[int, int, int]) -> int:
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  

        adv = F.relu(self.advantage_fc1(x))
        adv = self.advantage_fc2(adv)

        val = F.relu(self.value_fc1(x))
        val = self.value_fc2(val)
        
        
        
        return val + adv - adv.mean(dim=1, keepdim=True)

class ReplayBuffer:
    def __init__(self, capacity: int, frame_stack_size: int, image_dims: Tuple[int, int]):
        self.capacity = capacity
        self.frame_stack_size = frame_stack_size
        self.image_height, self.image_width = image_dims
        
        
        self.memory = collections.deque(maxlen=capacity)
        self.frame_dtype = np.uint8 
        
        
        
        

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(self.memory) < batch_size or len(self.memory) < self.frame_stack_size:
            return None

        indices = random.sample(range(self.frame_stack_size -1, len(self.memory)), batch_size)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for i in indices:
            
            current_state_stack = self._get_stacked_frames(i)
            
            
            
            
            
            
            
            
            state_tuple = self.memory[i]
            action, reward, next_single_frame, done = state_tuple[1], state_tuple[2], state_tuple[3], state_tuple[4]

            
            
            
            
            if done:
                
                next_state_stack = np.zeros_like(current_state_stack)
            else:
                
                
                next_state_stack = self._get_stacked_frames_for_next_state(i)


            batch_states.append(current_state_stack)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_next_states.append(next_state_stack)
            batch_dones.append(done)

        
        
        states_tensor = torch.from_numpy(np.array(batch_states)).float() / 255.0
        actions_tensor = torch.tensor(batch_actions, dtype=torch.int64).unsqueeze(1) 
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1) 
        next_states_tensor = torch.from_numpy(np.array(batch_next_states)).float() / 255.0
        dones_tensor = torch.tensor(batch_dones, dtype=torch.bool).unsqueeze(1) 
        
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def _get_stacked_frames(self, index: int) -> np.ndarray:
        
        
        
        frames = []
        for i in range(self.frame_stack_size):
            
            frame_index = index - (self.frame_stack_size - 1) + i
            if frame_index < 0: 
                frames.append(np.zeros((self.image_height, self.image_width), dtype=self.frame_dtype))
            else:
                
                
                
                
                frames.append(self.memory[frame_index][0]) 
        
        
        return np.stack(frames, axis=0)

    def _get_stacked_frames_for_next_state(self, current_index: int) -> np.ndarray:
        
        
        
        
        frames = []
        
        frames.append(self.memory[current_index][3]) 

        
        
        for i in range(1, self.frame_stack_size):
            frame_lookback_index = current_index - (i - 1)
            if frame_lookback_index < 0:
                frames.insert(0, np.zeros((self.image_height, self.image_width), dtype=self.frame_dtype))
            else:
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

                
                
                idx_for_prev_frame = current_index - i + 1 
                if idx_for_prev_frame < 0: 
                     frames.insert(0, np.zeros((self.image_height, self.image_width), dtype=self.frame_dtype))
                else:
                     frames.insert(0, self.memory[idx_for_prev_frame][0]) 

        return np.stack(frames, axis=0)


    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(self, model_config: Dict[str, Any], device: torch.device):
        self.device = device
        self.num_actions = model_config["num_actions"]
        self.gamma = model_config["gamma"]
        self.epsilon = model_config["epsilon_start"]
        self.epsilon_start = model_config["epsilon_start"]
        self.epsilon_end = model_config["epsilon_end"]
        self.epsilon_decay = model_config["epsilon_decay"]
        self.target_update_frequency = model_config["target_update_frequency"]
        self.save_path = model_config["save_path"]
        
        self.image_height = model_config["image_height"]
        self.image_width = model_config["image_width"]
        self.frame_stack = model_config["frame_stack"] 

        input_shape = (self.frame_stack, self.image_height, self.image_width)

        self.policy_net = DuelingCNN(input_shape, self.num_actions).to(self.device)
        self.target_net = DuelingCNN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=model_config["learning_rate"], amsgrad=True)
        self.total_steps = 0
        
        self.load_model(self.save_path)


    def select_action(self, state_stack: np.ndarray, is_eval: bool = False) -> int:
        self.total_steps += 1
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * self.total_steps / self.epsilon_decay)
        self.epsilon = max(self.epsilon_end, self.epsilon) 

        if is_eval or random.random() > self.epsilon:
            with torch.no_grad():
                
                
                state_tensor = torch.from_numpy(state_stack).unsqueeze(0).float().to(self.device) / 255.0
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item() 
        else:
            return random.randrange(self.num_actions)

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Optional[float]:
        if len(replay_buffer) < model_config.get("min_replay_buffer_size", batch_size) :
            return None

        batch = replay_buffer.sample(batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        
        
        current_q_values = self.policy_net(states).gather(1, actions)

        
        
        
        
        with torch.no_grad():
            next_policy_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_target_q_values = self.target_net(next_states).gather(1, next_policy_actions)
        
        
        next_target_q_values[dones] = 0.0
        
        
        expected_q_values = rewards + (self.gamma * next_target_q_values)

        
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()

        
        if self.total_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logging.info("Updated target network.")

        return loss.item()

    def save_model(self, path: Optional[str] = None):
        save_to = path if path else self.save_path
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'epsilon': self.epsilon
            }, save_to)
            logging.info(f"Model saved to {save_to}")
        except Exception as e:
            logging.error(f"Error saving model to {save_to}: {e}")

    def load_model(self, path: Optional[str] = None):
        load_from = path if path else self.save_path
        if os.path.exists(load_from):
            try:
                checkpoint = torch.load(load_from, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.total_steps = checkpoint.get('total_steps', 0) 
                self.epsilon = checkpoint.get('epsilon', self.epsilon_start) 
                self.policy_net.train() 
                self.target_net.eval()  
                logging.info(f"Model loaded from {load_from}. Total steps: {self.total_steps}, Epsilon: {self.epsilon:.4f}")
            except Exception as e:
                logging.error(f"Error loading model from {load_from}: {e}. Starting with a fresh model.")
        else:
            logging.info(f"No model found at {load_from}. Starting with a fresh model.")

    @staticmethod
    def preprocess_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_resized = cv2.resize(frame_gray, (width, height), interpolation=cv2.INTER_AREA)
        
        return frame_resized



class GameController:
    def __init__(self, config_manager: ConfigManager):
        self.mouse = MouseController()
        self.game_config = config_manager.get_section("game")
        self.death_config = config_manager.get_section("death_detection")
        self.restart_cooldown_s = self.game_config.get("restart_cooldown_ms", 1000) / 1000.0
        self.last_death_time = 0
        
        self.death_roi_config = self.death_config.get("death_detection_roi", {})
        self.use_roi_death_detection = self.death_roi_config.get("use", False) if self.death_roi_config else False
        
        if self.use_roi_death_detection and self.death_roi_config:
            self.roi_x_ratio = self.death_roi_config.get("x_start_ratio", 0.0)
            self.roi_y_ratio = self.death_roi_config.get("y_start_ratio", 0.0)
            self.roi_w_ratio = self.death_roi_config.get("width_ratio", 0.1)
            self.roi_h_ratio = self.death_roi_config.get("height_ratio", 0.1)
            
            
            
            
        
        self.previous_frame_gray_simple = None 

    def jump(self):
        self.mouse.press(MouseButton.left)
        time.sleep(0.05) 
        self.mouse.release(MouseButton.left)
        logging.debug("Action: Jump")

    def no_op(self):
        logging.debug("Action: No-op")
        pass

    def _is_player_dead_roi(self, current_frame_bgr: np.ndarray) -> bool:
        """ Rudimentary death detection based on ROI color or significant change. """
        if not self.use_roi_death_detection or not self.death_roi_config:
            return False

        h, w, _ = current_frame_bgr.shape
        roi_x = int(w * self.roi_x_ratio)
        roi_y = int(h * self.roi_y_ratio)
        roi_w = int(w * self.roi_w_ratio)
        roi_h = int(h * self.roi_h_ratio)

        
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, w - roi_x)
        roi_h = min(roi_h, h - roi_y)

        if roi_w <=0 or roi_h <=0:
            logging.warning("Death detection ROI has zero or negative dimensions.")
            return False

        roi_frame = current_frame_bgr[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        
        
        
        
        
        
        

        
        
        gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_roi)
        
        if std_dev < 10: 
             logging.debug(f"Death detected: ROI very uniform (std_dev: {std_dev:.2f}).")
             return True
        
        return False
    
    def _is_player_dead_simple_change(self, current_frame_bgr: np.ndarray) -> bool:
        """ Very basic: checks for massive screen change, indicating potential death/reset. """
        
        
        
        
        gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame_gray_simple is None:
            self.previous_frame_gray_simple = gray
            return False

        frame_delta = cv2.absdiff(self.previous_frame_gray_simple, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1] 
        
        
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        change_percentage = (changed_pixels / total_pixels) * 100

        self.previous_frame_gray_simple = gray

        
        
        
        
        
        if change_percentage > 70: 
            logging.debug(f"Potential death/reset detected by simple screen change ({change_percentage:.2f}%).")
            return True
        return False


    def is_player_dead(self, current_frame_bgr: np.ndarray) -> bool:
        
        if self.use_roi_death_detection and self._is_player_dead_roi(current_frame_bgr):
            return True
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        return False 

    def restart_level(self) -> bool:
        current_time = time.time()
        if current_time - self.last_death_time < self.restart_cooldown_s:
            return False 

        
        
        
        
        logging.info("Attempting to restart level.")
        self.mouse.press(MouseButton.left)
        time.sleep(0.05)
        self.mouse.release(MouseButton.left)
        self.last_death_time = current_time
        self.previous_frame_gray_simple = None 
        return True



class Visualizer:
    def __init__(self, config_manager: ConfigManager):
        self.viz_config = config_manager.get_section("visualization")
        self.show_ai_vision = self.viz_config.get("show_ai_vision", True)
        self.show_training_plot = self.viz_config.get("show_training_plot", True)
        self.plot_update_freq = self.viz_config.get("plot_update_frequency_episodes", 1)
        self.stats_update_freq_s = self.viz_config.get("stats_update_frequency_seconds", 1)
        
        self.loss_history = []
        self.reward_history = [] 
        self.accuracy_history = [] 
        self.episode_count_for_plot = 0
        self.last_stats_update_time = time.time()

        if self.show_training_plot:
            plt.ion() 
            self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6)) 
            self.fig.canvas.manager.set_window_title("GDint Training Progress")


    def display_ai_vision(self, frame: np.ndarray, activation_map: Optional[np.ndarray] = None):
        if not self.show_ai_vision:
            return

        display_frame = frame.copy()
        if activation_map is not None:
            
            
            
            
            pass 

        cv2.imshow("GDint AI Vision", display_frame)
        

    def update_training_graph(self, episode: int, loss: Optional[float], total_reward: float):
        if not self.show_training_plot:
            return

        if loss is not None:
            self.loss_history.append(loss)
        self.reward_history.append(total_reward)
        

        self.episode_count_for_plot +=1

        if self.episode_count_for_plot % self.plot_update_freq == 0:
            self.axs[0].clear()
            self.axs[0].plot(self.loss_history, label="Loss")
            self.axs[0].set_title("Training Loss")
            self.axs[0].set_xlabel("Training Steps (or Batches)")
            self.axs[0].set_ylabel("Loss")
            self.axs[0].legend()

            self.axs[1].clear()
            self.axs[1].plot(self.reward_history, label="Total Reward per Episode")
            self.axs[1].set_title("Episode Rewards")
            self.axs[1].set_xlabel("Episode")
            self.axs[1].set_ylabel("Total Reward")
            
            if len(self.reward_history) >= 10:
                moving_avg = np.convolve(self.reward_history, np.ones(10)/10, mode='valid')
                self.axs[1].plot(np.arange(9, len(self.reward_history)), moving_avg, label="10-ep Moving Avg")
            self.axs[1].legend()
            
            plt.tight_layout()
            plt.pause(0.01) 

    def display_stats(self, stats: Dict[str, Any]):
        
        current_time = time.time()
        if current_time - self.last_stats_update_time > self.stats_update_freq_s:
            log_message_parts = []
            for key, value in stats.items():
                if isinstance(value, float):
                    log_message_parts.append(f"{key}: {value:.2f}")
                else:
                    log_message_parts.append(f"{key}: {value}")
            logging.info(f"Stats: {', '.join(log_message_parts)}")
            self.last_stats_update_time = current_time
            
            
            if self.show_ai_vision and cv2.getWindowProperty("GDint AI Vision", cv2.WND_PROP_VISIBLE) >=1:
                
                
                pass
                
    def close(self):
        if self.show_ai_vision:
            cv2.destroyWindow("GDint AI Vision")
        if self.show_training_plot:
            plt.ioff()
            plt.close(self.fig)


def toggle_pause():
    global bot_paused
    bot_paused = not bot_paused
    status = "PAUSED" if bot_paused else "RESUMED"
    logging.info(f"Bot operation {status}.")

def signal_exit():
    global bot_running
    bot_running = False
    logging.info("Exit signal received. Shutting down...")

def setup_hotkeys(config_manager: ConfigManager):
    hotkey_config = config_manager.get_section("hotkeys")
    pause_key = hotkey_config.get("pause_resume", "f8")
    exit_key = hotkey_config.get("exit_bot", "f9")

    try:
        keyboard.add_hotkey(pause_key, toggle_pause)
        keyboard.add_hotkey(exit_key, signal_exit)
        logging.info(f"Hotkeys registered: Pause/Resume ({pause_key}), Exit ({exit_key})")
    except Exception as e:
        logging.error(f"Failed to register hotkeys: {e}. Try running with administrator privileges.")



class GDintBot:
    def __init__(self):
        self.config_manager = ConfigManager()
        setup_logging(self.config_manager)
        setup_hotkeys(self.config_manager)

        self.capture = ScreenCapture(self.config_manager)
        self.controller = GameController(self.config_manager)
        self.visualizer = Visualizer(self.config_manager)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        global model_config 
        model_config = self.config_manager.get_section("model")
        self.agent = DQNAgent(model_config, self.device)
        
        self.replay_buffer = ReplayBuffer(
            capacity=model_config["replay_buffer_size"],
            frame_stack_size=model_config["frame_stack"],
            image_dims=(model_config["image_height"], model_config["image_width"])
        )
        
        self.frame_stack_size = model_config["frame_stack"]
        self.img_h = model_config["image_height"]
        self.img_w = model_config["image_width"]
        
        self.preprocessed_frame_buffer = collections.deque(maxlen=self.frame_stack_size)

        self.episode_count = 0
        self.total_steps_session = 0 
        self.loop_start_time = time.time()
        self.frames_processed_for_fps = 0


    def _get_current_state_stack(self) -> Optional[np.ndarray]:
        if len(self.preprocessed_frame_buffer) < self.frame_stack_size:
            
            
            padded_stack = []
            num_missing = self.frame_stack_size - len(self.preprocessed_frame_buffer)
            for _ in range(num_missing):
                padded_stack.append(np.zeros((self.img_h, self.img_w), dtype=np.uint8))
            for frame in self.preprocessed_frame_buffer:
                padded_stack.append(frame)
            return np.stack(padded_stack, axis=0) if padded_stack else None
        return np.stack(list(self.preprocessed_frame_buffer), axis=0)

    def _add_frame_to_buffer(self, processed_frame: np.ndarray):
        self.preprocessed_frame_buffer.append(processed_frame)

    def _reset_episode_state(self):
        self.preprocessed_frame_buffer.clear()
        
        logging.info(f"Episode {self.episode_count} ended. Resetting frame buffer.")


    def run(self):
        global bot_running, bot_paused

        if not self.capture.find_and_focus_window():
            logging.error("Failed to initialize game window. Exiting.")
            return

        last_frame_time = time.time()
        
        current_episode_reward = 0.0
        current_episode_steps = 0
        
        
        for _ in range(self.frame_stack_size): 
            initial_raw_frame = self.capture.get_frame()
            if initial_raw_frame is None:
                logging.error("Could not get initial frames. Exiting.")
                bot_running = False 
                break
            processed = DQNAgent.preprocess_frame(initial_raw_frame, self.img_h, self.img_w)
            self._add_frame_to_buffer(processed)
        
        if not bot_running: 
             self.cleanup()
             return

        while bot_running:
            if bot_paused:
                time.sleep(0.1) 
                
                last_frame_time = time.time() 
                self.loop_start_time = time.time()
                self.frames_processed_for_fps = 0
                continue

            
            loop_delta = time.time() - self.loop_start_time
            if loop_delta >= 1.0: 
                capture_fps = self.frames_processed_for_fps / loop_delta
                self.visualizer.display_stats({
                    "Capture FPS": capture_fps,
                    "Episode": self.episode_count,
                    "Session Steps": self.total_steps_session,
                    "Agent Epsilon": self.agent.epsilon,
                    "Replay Buffer Size": len(self.replay_buffer)
                })
                self.loop_start_time = time.time()
                self.frames_processed_for_fps = 0
            
            time_since_last_frame = time.time() - last_frame_time
            if time_since_last_frame < self.capture.frame_interval:
                time.sleep(self.capture.frame_interval - time_since_last_frame)
            last_frame_time = time.time()

            
            raw_frame = self.capture.get_frame()
            if raw_frame is None:
                logging.warning("Failed to capture frame, skipping step.")
                time.sleep(0.5) 
                if not self.capture.find_and_focus_window(): 
                    logging.error("Lost game window and cannot re-acquire. Exiting.")
                    bot_running = False
                continue
            
            self.frames_processed_for_fps += 1

            
            processed_frame = DQNAgent.preprocess_frame(raw_frame, self.img_h, self.img_w)
            
            
            current_state_stack = self._get_current_state_stack() 
            if current_state_stack is None: 
                logging.warning("Current state stack is None, skipping.")
                self._add_frame_to_buffer(processed_frame) 
                continue
                
            self._add_frame_to_buffer(processed_frame) 

            
            next_state_stack = self._get_current_state_stack() 
            if next_state_stack is None: 
                 logging.warning("Next state stack is None, this is an error.")
                 
                 
                 
                 pass


            
            action = self.agent.select_action(current_state_stack)

            
            if action == 1: 
                self.controller.jump()
            else: 
                self.controller.no_op()

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

            
            
            
            
            
            
            
            
            
            is_dead = self.controller.is_player_dead(raw_frame) 
            reward = 0.0
            done_flag = False

            if is_dead:
                reward = -100.0 
                done_flag = True
                logging.info(f"Player died. Episode Reward: {current_episode_reward + reward}")
                self.controller.restart_level() 
                time.sleep(self.controller.restart_cooldown_s) 
            else:
                reward = 0.1 
            
            current_episode_reward += reward
            current_episode_steps += 1
            self.total_steps_session += 1
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            if hasattr(self, 'last_step_data_for_buffer') and self.last_step_data_for_buffer:
                prev_processed_state, prev_action, prev_reward, prev_done = self.last_step_data_for_buffer
                
                self.replay_buffer.push(prev_processed_state, prev_action, prev_reward, processed_frame, prev_done)

            
            self.last_step_data_for_buffer = (processed_frame, action, reward, done_flag)


            
            if self.total_steps_session % model_config.get("train_frequency_steps", 4) == 0: 
                loss = self.agent.train_step(self.replay_buffer, model_config["batch_size"])
                if loss is not None and self.episode_count > 0 : 
                    
                    
                    
                    
                    
                    
                    
                    if not hasattr(self, 'current_episode_losses'):
                        self.current_episode_losses = []
                    self.current_episode_losses.append(loss)


            
            if self.visualizer.show_ai_vision:
                
                self.visualizer.display_ai_vision(raw_frame, activation_map=None)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                if self.visualizer.show_ai_vision: 
                     bot_running = False


            
            if done_flag:
                self.episode_count += 1
                avg_loss_this_episode = np.mean(self.current_episode_losses) if hasattr(self, 'current_episode_losses') and self.current_episode_losses else None
                self.visualizer.update_training_graph(self.episode_count, avg_loss_this_episode, current_episode_reward)
                
                logging.info(f"Episode {self.episode_count} finished. Steps: {current_episode_steps}, Total Reward: {current_episode_reward:.2f}, Avg Loss: {avg_loss_this_episode if avg_loss_this_episode else 'N/A'}")
                
                
                current_episode_reward = 0.0
                current_episode_steps = 0
                if hasattr(self, 'current_episode_losses'): self.current_episode_losses.clear()
                self._reset_episode_state() 
                
                
                for _ in range(self.frame_stack_size):
                    initial_raw_frame_ep = self.capture.get_frame()
                    if initial_raw_frame_ep is None:
                        logging.error("Could not get initial frames for new episode. Exiting.")
                        bot_running = False; break
                    processed_ep = DQNAgent.preprocess_frame(initial_raw_frame_ep, self.img_h, self.img_w)
                    self._add_frame_to_buffer(processed_ep)
                if not bot_running: break 

                
                if hasattr(self, 'last_step_data_for_buffer'):
                    del self.last_step_data_for_buffer


                
                if self.episode_count % model_config.get("save_model_frequency_episodes", 10) == 0:
                    self.agent.save_model()
            
            if not bot_running: 
                break


        self.cleanup()

    def cleanup(self):
        logging.info("Cleaning up and exiting...")
        self.agent.save_model() 
        self.visualizer.close()
        if self.capture.sct:
            self.capture.sct.close()
        cv2.destroyAllWindows()
        keyboard.unhook_all() 
        logging.info("GDint bot stopped.")

if __name__ == "__main__":
    gd_bot = GDintBot()
    try:
        gd_bot.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down.")
        gd_bot.cleanup()
    except Exception as e:
        logging.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        gd_bot.cleanup()