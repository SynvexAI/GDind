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
SHOW_GAME_REGION_OUTLINE = True
GAME_REGION_OUTLINE_BORDER_COLOR = "lime"
GAME_REGION_OUTLINE_THICKNESS = 2

# --- GUI ---
ENABLE_GUI = True
GUI_UPDATE_INTERVAL_MS = 30
GUI_SHOW_RAW_CAPTURE = True
GUI_RAW_CAPTURE_DISPLAY_SCALE = 0.5
GUI_AI_VIEW_DISPLAY_SCALE = 3

# --- Захват игры ---
FALLBACK_GAME_REGION = None
AI_FPS_LIMIT = 60  # Умеренный FPS для стабильности

# --- Обработка изображений ---
FRAME_WIDTH = 96
FRAME_HEIGHT = 96
GRAYSCALE = True
NUM_FRAMES_STACKED = 4

# --- Модель ИИ (DQN) ---
LEARNING_RATE = 0.0002  # Чуть ниже для стабильности
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_FRAMES = 50000  # Более плавное снижение epsilon

# --- Обучение ---
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 80000
TARGET_UPDATE_FREQ_EPISODES = 4
LEARN_START_STEPS = 3000
MAX_STEPS_PER_EPISODE = 1500
NUM_EPISODES = 15000

# --- Действия ---
NUM_ACTIONS = 2
ACTION_DELAY = 0.015
JUMP_DURATION = 0.07

# --- Обнаружение "Game Over" ---
GAME_OVER_TEMPLATE_PATH = "game_over_template.png"
GAME_OVER_DETECTION_THRESHOLD = 0.75
GAME_OVER_SEARCH_REGION = None

# --- Награды ---
REWARD_ALIVE = 0.01
REWARD_DEATH = -1.0
REWARD_PROGRESS_FACTOR = 0.6

# --- Сохранения ---
MODEL_SAVE_PATH = f"{PROJECT_NAME.lower()}_model.pth"
SAVE_MODEL_EVERY_N_EPISODES = 20
PLOT_SAVE_PATH = f"{PROJECT_NAME.lower()}_training_plots.png"


SHOW_GAME_REGION_OUTLINE      = False
GUI_SHOW_RAW_CAPTURE         = False
GUI_MARK_DETECTED_OBJECTS    = False
