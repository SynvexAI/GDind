import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import cv2
import numpy as np
import mss
import time
import random
from collections import deque, namedtuple
from pynput.mouse import Button, Controller as MouseController

import config

if torch.cuda.is_available() and config.DEVICE == "cuda":
    DEVICE = torch.device("cuda")
    print("Using CUDA (GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")
    config.DEVICE = "cpu" 

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


sct = mss.mss()
mouse = MouseController()

def get_screen_region():
    if config.GAME_REGION:
        monitor = config.GAME_REGION
    else:
        
        monitor = sct.monitors[1] 
    return monitor

monitor_region = get_screen_region()

def capture_screen():
    sct_img = sct.grab(monitor_region)
    img = np.array(sct_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) 
    return img

def preprocess_image(image):
    if config.GRAYSCALE:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    if config.GRAYSCALE:
        image = np.expand_dims(image, axis=0) 
    else:
        image = np.transpose(image, (2, 0, 1)) 
    image = image.astype(np.float32) / 255.0 
    return torch.from_numpy(image).unsqueeze(0).to(DEVICE) 


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        
        in_channels = 1 if config.GRAYSCALE else 3

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



class Agent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.policy_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions).to(DEVICE)
        self.target_net = DQN(config.FRAME_HEIGHT, config.FRAME_WIDTH, num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=config.LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(config.REPLAY_MEMORY_SIZE)
        self.steps_done = 0
        self.epsilon = config.EPSILON_START

    def select_action(self, state):
        self.epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * \
                       np.exp(-1. * self.steps_done / config.EPSILON_DECAY)
        self.steps_done += 1
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=DEVICE, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < config.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)

        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)    
        
        next_state_values = torch.zeros(config.BATCH_SIZE, device=DEVICE)
        with torch.no_grad(): 
             next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        
        expected_state_action_values = (next_state_values * config.GAMMA * (1-done_batch.float())) + reward_batch


        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()


    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="geometry_dash_ai_model.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="geometry_dash_ai_model.pth"):
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.target_net.load_state_dict(self.policy_net.state_dict()) 
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Model file not found at {path}, starting with a new model.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting with a new model.")



def perform_action(action_tensor):
    action = action_tensor.item() 
    if action == 1: 
        print("Action: JUMP")
        mouse.press(Button.left)
        time.sleep(config.JUMP_DURATION) 
        mouse.release(Button.left)
    else: 
        
        pass






def get_reward_and_done_status(current_screen_pil_image):

    return 1.0, False 



def main():
    print("Starting AI for Geometry Dash...")
    print(f"Using device: {DEVICE}")
    print(f"Game region: {monitor_region}")
    print("Ensure the Geometry Dash window is active and visible in the specified region.")
    print("Press Ctrl+C in the console to stop.")
    
    
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    agent = Agent(num_actions=config.NUM_ACTIONS)
    
    agent.load_model()


    total_rewards = []
    losses = []

    try:
        for i_episode in range(config.NUM_EPISODES):
            print(f"\n--- Episode {i_episode + 1}/{config.NUM_EPISODES} ---")
            print(f"Current Epsilon: {agent.epsilon:.4f}")

            current_frame_raw = capture_screen()
            state = preprocess_image(current_frame_raw)
            
            episode_reward = 0
            episode_loss_sum = 0
            episode_steps = 0

            for t in range(config.MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state)
                perform_action(action)
                
                time.sleep(config.ACTION_DELAY) 

                next_frame_raw = capture_screen()
          
                reward_val = 1.0 
                done_val = False 
                
                episode_reward += reward_val
                
                reward = torch.tensor([reward_val], device=DEVICE, dtype=torch.float)
                done_tensor = torch.tensor([done_val], device=DEVICE, dtype=torch.bool) 

                if done_val:
                    next_state = None
                else:
                    next_state = preprocess_image(next_frame_raw)

                agent.memory.push(state, action, next_state, reward, done_tensor)
                state = next_state
                
                current_loss = agent.optimize_model()
                if current_loss is not None:
                    episode_loss_sum += current_loss
                    episode_steps +=1

                if done_val or t == config.MAX_STEPS_PER_EPISODE - 1:
                    print(f"Episode finished after {t+1} steps. Reward: {episode_reward:.2f}")
                    if episode_steps > 0:
                         print(f"Average Loss: {episode_loss_sum / episode_steps:.4f}")
                    break
            
            total_rewards.append(episode_reward)
            if episode_steps > 0 : losses.append(episode_loss_sum / episode_steps)


            if (i_episode + 1) % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
                print("Updated target network.")

            
            if (i_episode + 1) % (config.TARGET_UPDATE_FREQ * 5) == 0: 
                 agent.save_model()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving final model...")
        agent.save_model()
        print("Training finished.")
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        

if __name__ == '__main__':
    main()