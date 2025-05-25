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
    logging.info(f"Logging setup complete. Log level: {config.LOG_LEVEL.upper()}")
    logging.info(f"Using device: {config.DEVICE}")

setup_logging()



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        logging.info(f"ReplayMemory initialized with capacity: {capacity}")

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)
        logging.info(f"DQN model initialized. Input: ({in_channels}, {h}, {w}), Output: {outputs}, Linear input size: {linear_input_size}")

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
        self.monitor_region = self._get_screen_region()
        logging.info(f"GameEnvironment initialized. Screen region: {self.monitor_region}")

        if config.GAME_OVER_TEMPLATE_PATH and os.path.exists(config.GAME_OVER_TEMPLATE_PATH):
            self.game_over_template = cv2.imread(config.GAME_OVER_TEMPLATE_PATH)
            if config.GRAYSCALE:
                self.game_over_template = cv2.cvtColor(self.game_over_template, cv2.COLOR_BGR2GRAY)
            logging.info(f"Loaded game over template from: {config.GAME_OVER_TEMPLATE_PATH}")
            if config.GAME_OVER_SEARCH_REGION:
                logging.info(f"Game over search region: {config.GAME_OVER_SEARCH_REGION}")
        else:
            self.game_over_template = None
            logging.warning(f"Game over template not found or path not specified: {config.GAME_OVER_TEMPLATE_PATH}. Death detection will be basic.")

        
        self.stacked_frames = deque(maxlen=config.NUM_FRAMES_STACKED)
        self.frame_skip = 1 

        if config.DEBUG_VISUALIZE_INPUT:
            cv2.namedWindow("AI Input", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AI Input",
                             config.FRAME_WIDTH * config.VISUALIZATION_WINDOW_SCALE,
                             config.FRAME_HEIGHT * config.VISUALIZATION_WINDOW_SCALE * (1 if config.GRAYSCALE else config.NUM_FRAMES_STACKED) )


    def _get_screen_region(self):
        if config.GAME_REGION:
            return config.GAME_REGION
        try:
            return self.sct.monitors[1] 
        except IndexError:
            logging.error("Could not find primary monitor. Falling back to all screens.")
            return self.sct.monitors[0] 

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
        if config.GRAYSCALE:
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            frame = frame_bgr 
        
        frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        
        if config.GRAYSCALE:
             
            return frame_resized.astype(np.uint8) 
        else:
             
            return np.transpose(frame_resized, (2, 0, 1)).astype(np.uint8)


    def _stack_frames(self, processed_frame):
        if not self.stacked_frames: 
            for _ in range(config.NUM_FRAMES_STACKED):
                self.stacked_frames.append(processed_frame)
        else:
            self.stacked_frames.append(processed_frame)
        
        
        if config.GRAYSCALE:           
            stacked_state = np.array(self.stacked_frames) 
        else:
            
            
            stacked_state = np.concatenate(list(self.stacked_frames), axis=0) 

        return torch.from_numpy(stacked_state).unsqueeze(0).to(config.DEVICE).float() 

    def reset(self):
        logging.debug("Resetting environment.")                            

        self.stacked_frames.clear()
        raw_frame = self._capture_frame()
        processed_frame = self._preprocess_frame(raw_frame)
        
        for _ in range(config.NUM_FRAMES_STACKED): 
             self.stacked_frames.append(processed_frame)
        
        current_state = self._stack_frames(processed_frame) 
        return current_state, raw_frame 

    def step(self, action_tensor):
        action = action_tensor.item()
        if action == 1: 
            logging.debug("Action: JUMP")
            self.mouse.press(Button.left)
            time.sleep(config.JUMP_DURATION)
            self.mouse.release(Button.left)
        

        if config.ACTION_DELAY > 0:
            time.sleep(config.ACTION_DELAY)

        raw_next_frame = self._capture_frame()
        processed_next_frame = self._preprocess_frame(raw_next_frame)
        next_state_tensor = self._stack_frames(processed_next_frame) 

        reward, done = self._get_reward_and_done(raw_next_frame)

        if config.DEBUG_VISUALIZE_INPUT:
            
            
            display_frame = self.stacked_frames[-1] 
            if config.GRAYSCALE:
                 
                 display_frame_resized = cv2.resize(display_frame,
                                                (config.FRAME_WIDTH * config.VISUALIZATION_WINDOW_SCALE,
                                                config.FRAME_HEIGHT * config.VISUALIZATION_WINDOW_SCALE),
                                                interpolation=cv2.INTER_NEAREST) 
                 cv2.imshow("AI Input", display_frame_resized)

            else: 
                
                display_frame_chw = np.transpose(display_frame, (1,2,0))
                display_frame_resized = cv2.resize(display_frame_chw,
                                                (config.FRAME_WIDTH * config.VISUALIZATION_WINDOW_SCALE,
                                                config.FRAME_HEIGHT * config.VISUALIZATION_WINDOW_SCALE),
                                                interpolation=cv2.INTER_NEAREST)
                cv2.imshow("AI Input", display_frame_resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                 logging.info("Visualization window closed by user ('q' pressed).")
                 


        return next_state_tensor, reward, done, raw_next_frame


    def _get_reward_and_done(self, current_raw_frame):
        """
        Определяет награду и флаг завершения эпизода.
        ЭТО КЛЮЧЕВАЯ ЧАСТЬ, КОТОРУЮ НУЖНО АДАПТИРОВАТЬ!
        """
        done = False
        reward = config.REWARD_ALIVE

        
        if self.game_over_template is not None:
            frame_to_search = current_raw_frame
            if config.GRAYSCALE:
                frame_to_search = cv2.cvtColor(current_raw_frame, cv2.COLOR_BGR2GRAY)

            search_area = frame_to_search 
            if config.GAME_OVER_SEARCH_REGION:
                x, y, w, h = config.GAME_OVER_SEARCH_REGION
                search_area = frame_to_search[y:y+h, x:x+w]
            
            if search_area.shape[0] < self.game_over_template.shape[0] or \
               search_area.shape[1] < self.game_over_template.shape[1]:
                
                pass 
            else:
                res = cv2.matchTemplate(search_area, self.game_over_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > config.GAME_OVER_DETECTION_THRESHOLD:
                    logging.info(f"Game Over detected (match: {max_val:.2f} > {config.GAME_OVER_DETECTION_THRESHOLD}).")
                    done = True
                    reward = config.REWARD_DEATH
                    if config.SAVE_FRAMES_ON_DEATH:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(f"death_frame_{timestamp}.png", current_raw_frame)
                        logging.debug(f"Saved death frame: death_frame_{timestamp}.png")
        
        
        
        

        
        return reward, done



class Agent:
    def __init__(self, num_actions, env_for_shape_calc):
        self.num_actions = num_actions
        
        
        initial_state, _ = env_for_shape_calc.reset() 
        _, c, h, w = initial_state.shape 
        
        
        
        
        num_input_channels = (config.NUM_FRAMES_STACKED if config.GRAYSCALE else 3 * config.NUM_FRAMES_STACKED)

        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, num_frames=config.NUM_FRAMES_STACKED).to(config.DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions, num_frames=config.NUM_FRAMES_STACKED).to(config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE) 
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.total_steps_done = 0 

        logging.info("Agent initialized.")
        logging.info(f"Policy Net: {self.policy_net}")


    def select_action(self, state_tensor): 
        self.total_steps_done += 1
        epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                       np.exp(-1. * self.total_steps_done / config.EPSILON_DECAY_FRAMES)
        
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].view(1, 1)
                logging.debug(f"Greedy action. Q-values: {q_values.cpu().numpy().round(2)}, Chosen action: {action.item()}")
                return action
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], device=config.DEVICE, dtype=torch.long)
            logging.debug(f"Random action (epsilon={epsilon:.4f}). Chosen action: {action.item()}")
            return action

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE or self.total_steps_done < config.LEARN_START_STEPS:
            return None 
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=config.DEVICE, dtype=torch.bool)
        
        if any(non_final_mask):
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        else: 
            non_final_next_states = None 

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(config.BATCH_SIZE, device=config.DEVICE)
        if non_final_next_states is not None and non_final_next_states.size(0) > 0 : 
            with torch.no_grad():
                 next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * config.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss() 
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        logging.debug(f"Optimization step. Loss: {loss.item():.4f}")
        return loss.item()

    def update_target_net(self):
        logging.info("Updating target network.")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path=config.MODEL_SAVE_PATH):
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'total_steps_done': self.total_steps_done,
                
            }, path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")


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

            self.policy_net.to(config.DEVICE) 
            self.target_net.to(config.DEVICE)
            self.target_net.eval() 

            
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(config.DEVICE)

            logging.info(f"Model loaded from {path}. Resuming from {self.total_steps_done} total steps.")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}. Starting with a new model.")
            return False


def plot_training_results(episode_rewards, episode_losses, episode_durations, save_path=config.PLOT_SAVE_PATH):
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid') 

        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        
        axs[0].plot(episode_rewards, label='Total Reward per Episode', color='dodgerblue')
        axs[0].plot(pd.Series(episode_rewards).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 episodes)', color='orangered', linestyle='--')
        axs[0].set_ylabel('Total Reward')
        axs[0].legend()
        axs[0].set_title('Training Rewards')

        
        valid_losses = [l for l in episode_losses if l is not None]
        if valid_losses:
            axs[1].plot(valid_losses, label='Average Loss per Episode', color='mediumseagreen')
            axs[1].plot(pd.Series(valid_losses).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 episodes)', color='purple', linestyle='--')
        axs[1].set_ylabel('Average Loss')
        axs[1].legend()
        axs[1].set_title('Training Loss (SmoothL1)')
        
        
        axs[2].plot(episode_durations, label='Steps per Episode', color='goldenrod')
        axs[2].plot(pd.Series(episode_durations).rolling(50, min_periods=1).mean(), label='Rolling Mean (50 episodes)', color='saddlebrown', linestyle='--')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Number of Steps')
        axs[2].legend()
        axs[2].set_title('Episode Durations')

        fig.suptitle(f'{config.PROJECT_NAME} - Training Progress', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Training plots saved to {save_path}")
        

    except ImportError:
        logging.warning("Matplotlib or pandas not found. Skipping plot generation. Install with 'pip install matplotlib pandas'")
    except Exception as e:
        logging.error(f"Error plotting results: {e}")



def main():
    logging.info(f"Starting {config.PROJECT_NAME} AI...")
    logging.info("Ensure the Geometry Dash window is active and visible in the specified region.")
    logging.info("Press Ctrl+C in the console to stop.")
    
    
    global pd
    try:
        import pandas as pd
    except ImportError:
        pd = None


    for i in range(3, 0, -1):
        logging.info(f"Starting in {i}...")
        time.sleep(1)

    env = GameEnvironment()
    agent = Agent(num_actions=config.NUM_ACTIONS, env_for_shape_calc=env) 
    
    
    agent.load_model()

    all_episode_rewards = []
    all_episode_avg_losses = []
    all_episode_durations = []
    
    frame_limiter_delay = 1.0 / config.FPS_LIMIT if config.FPS_LIMIT > 0 else 0

    try:
        for i_episode in range(1, config.NUM_EPISODES + 1):
            loop_start_time = time.perf_counter()
            
            current_state_tensor, _ = env.reset() 
            episode_reward = 0
            episode_loss_sum = 0
            episode_optimization_steps = 0
            
            logging.info(f"--- Episode {i_episode}/{config.NUM_EPISODES} --- Epsilon: {config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * np.exp(-1. * agent.total_steps_done / config.EPSILON_DECAY_FRAMES):.4f}")

            for t_step in range(config.MAX_STEPS_PER_EPISODE):
                step_start_time = time.perf_counter()

                action_tensor = agent.select_action(current_state_tensor)
                next_state_tensor, reward_val, done, _ = env.step(action_tensor) 

                episode_reward += reward_val
                
                reward_tensor = torch.tensor([reward_val], device=config.DEVICE, dtype=torch.float)
                done_tensor = torch.tensor([done], device=config.DEVICE, dtype=torch.bool) 

                agent.memory.push(current_state_tensor, action_tensor, 
                                  None if done else next_state_tensor, 
                                  reward_tensor, done_tensor) 
                
                current_state_tensor = next_state_tensor 
                
                loss = agent.optimize_model()
                if loss is not None:
                    episode_loss_sum += loss
                    episode_optimization_steps += 1
                
                if frame_limiter_delay > 0:
                    elapsed_step_time = time.perf_counter() - step_start_time
                    sleep_time = frame_limiter_delay - elapsed_step_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if done:
                    break
            
            
            all_episode_rewards.append(episode_reward)
            all_episode_durations.append(t_step + 1)
            avg_loss = episode_loss_sum / episode_optimization_steps if episode_optimization_steps > 0 else None
            all_episode_avg_losses.append(avg_loss)

            logging.info(f"Episode {i_episode} finished. Steps: {t_step + 1}, Reward: {episode_reward:.2f}, Avg Loss: {avg_loss if avg_loss is not None else 'N/A'}")
            
            if i_episode % config.TARGET_UPDATE_FREQ_EPISODES == 0:
                agent.update_target_net()

            if i_episode % config.SAVE_MODEL_EVERY_N_EPISODES == 0:
                 agent.save_model()
                 if pd is not None: 
                    plot_training_results(all_episode_rewards, all_episode_avg_losses, all_episode_durations)
            
            loop_duration = time.perf_counter() - loop_start_time
            logging.debug(f"Episode {i_episode} total time: {loop_duration:.2f}s. Steps/sec: {(t_step+1)/loop_duration:.2f}")


    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during training: {e}", exc_info=True)
    finally:
        logging.info("Performing final cleanup and saving...")
        agent.save_model(f"{config.PROJECT_NAME.lower()}_final_model.pth")
        if cv2.getWindowProperty("AI Input", 0) >= 0 and config.DEBUG_VISUALIZE_INPUT: 
             cv2.destroyWindow("AI Input")
        if pd is not None:
            plot_training_results(all_episode_rewards, all_episode_avg_losses, all_episode_durations, f"{config.PROJECT_NAME.lower()}_final_plots.png")
        logging.info("Training finished.")


if __name__ == '__main__':
    main()