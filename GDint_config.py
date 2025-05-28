import torch

# --- Общие настройки ---
PROJECT_NAME = "GDint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_TITLE_SUBSTRING = "Geometry Dash"

# --- Управление ---
PAUSE_RESUME_KEY = 'p'

# --- Логирование и отладка ---
LOG_LEVEL = "INFO"
LOG_FILE = f"{PROJECT_NAME.lower()}_log.txt"
SAVE_FRAMES_ON_DEATH = True
SHOW_GAME_REGION_OUTLINE = True # Set to False if distracting, True for initial setup
GAME_REGION_OUTLINE_BORDER_COLOR = "lime"
GAME_REGION_OUTLINE_THICKNESS = 2

# --- GUI ---
ENABLE_GUI = True
GUI_UPDATE_INTERVAL_MS = 30  # Approx 33 FPS for GUI updates
GUI_SHOW_RAW_CAPTURE = True
GUI_RAW_CAPTURE_DISPLAY_SCALE = 0.4 # Adjusted for potentially larger GUI
GUI_AI_VIEW_DISPLAY_SCALE = 2.5 # Adjusted for potentially larger GUI

# --- Захват игры ---
FALLBACK_GAME_REGION = None # Example: {"top": 40, "left": 0, "width": 800, "height": 600}
AI_FPS_LIMIT = 60  # Target FPS for AI logic (capturing and processing)

# --- Обработка изображений ---
FRAME_WIDTH = 96   # Input width for the DQN
FRAME_HEIGHT = 96  # Input height for the DQN
GRAYSCALE = True
NUM_FRAMES_STACKED = 4 # Number of frames to stack as input to DQN

# --- Модель ИИ (DQN) ---
# Adjusted for a more powerful setup and longer training
LEARNING_RATE = 0.0001  # Slightly lower for larger batch/network
GAMMA = 0.99 # Discount factor for future rewards
EPSILON_START = 1.0 # Initial exploration rate
EPSILON_END = 0.02  # Final exploration rate
EPSILON_DECAY_FRAMES = 150000 # More gradual decay for longer training

# --- Обучение ---
# Adjusted for a more powerful setup
BATCH_SIZE = 128 # Increased batch size
REPLAY_MEMORY_SIZE = 150000 # Increased replay memory
TARGET_UPDATE_FREQ_EPISODES = 5 # How often to update the target network (in episodes)
LEARN_START_STEPS = 10000 # Number of steps to fill replay memory before starting learning
MAX_STEPS_PER_EPISODE = 2000 # Max steps per game attempt
NUM_EPISODES = 20000 # Total number of episodes to train for

# --- Действия ---
NUM_ACTIONS = 2 # 0: Idle, 1: Jump
ACTION_DELAY = 0.015 # Small delay after performing an action
JUMP_DURATION = 0.07 # How long the jump (mouse click) is held

# --- Обнаружение "Game Over" ---
# CRITICAL: Provide a clear, small, and distinctive image of the "Game Over" screen or a part of it.
# The effectiveness of automatic death detection depends heavily on this template.
GAME_OVER_TEMPLATE_PATH = "game_over_template.png"
GAME_OVER_DETECTION_THRESHOLD = 0.75 # Confidence threshold for template matching
# Optional: Define a smaller region within the game window to search for the game over template.
# Format: (x, y, width, height) relative to the game window. Example: (100, 100, 200, 100)
GAME_OVER_SEARCH_REGION = None

# --- Награды ---
REWARD_ALIVE = 0.01  # Small positive reward for each step the agent stays alive
REWARD_DEATH = -1.0  # Large negative reward for dying
# This factor is for advanced reward shaping within GameEnvironment._get_reward_and_done()
# e.g., if you implement custom logic to detect passing obstacles or score changes.
# Set to 0.0 if not implementing such custom logic.
REWARD_PROGRESS_FACTOR = 0.0

# --- Сохранения ---
MODEL_SAVE_PATH = f"{PROJECT_NAME.lower()}_model.pth"
SAVE_MODEL_EVERY_N_EPISODES = 50 # Save model more frequently during longer training
PLOT_SAVE_PATH = f"{PROJECT_NAME.lower()}_training_plots.png"

# Original values from user, kept for reference if needed to revert specific settings
# SHOW_GAME_REGION_OUTLINE      = False
# GUI_SHOW_RAW_CAPTURE         = False
# GUI_MARK_DETECTED_OBJECTS    = False
# REWARD_PROGRESS_FACTOR = 0.6 # This was the old one used in main loop, now set to 0.0