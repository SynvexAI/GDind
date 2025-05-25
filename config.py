import torch

# --- Общие настройки ---
PROJECT_NAME = "GDint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_TITLE_SUBSTRING = "Geometry Dash" # Часть заголовка окна игры

# --- Параметры логирования и отладки ---
LOG_LEVEL = "INFO"
LOG_FILE = f"{PROJECT_NAME.lower()}_log.txt"
SAVE_FRAMES_ON_DEATH = True
DEBUG_VISUALIZE_AI_INPUT_SEPARATELY = False
VISUALIZATION_WINDOW_SCALE = 3

# --- GUI Настройки ---
ENABLE_GUI = True
GUI_UPDATE_INTERVAL_MS = 100

# --- Параметры захвата экрана и игры ---
FALLBACK_GAME_REGION = None # {"top": 40, "left": 0, "width": 800, "height": 600}
FPS_LIMIT = 60

# --- Параметры обработки изображений ---
FRAME_WIDTH = 96
FRAME_HEIGHT = 96
GRAYSCALE = True
NUM_FRAMES_STACKED = 4

# --- Параметры модели ИИ (DQN) ---
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRAMES = 50000 # Увеличим, так как шагов может быть больше

# --- Параметры обучения ---
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 30000
TARGET_UPDATE_FREQ_EPISODES = 5
LEARN_START_STEPS = 2000
MAX_STEPS_PER_EPISODE = 1500
NUM_EPISODES = 5000

# --- Параметры действий ---
NUM_ACTIONS = 2
ACTION_DELAY = 0.05
JUMP_DURATION = 0.08

# --- Параметры определения состояния игры (Reward & Done) ---
GAME_OVER_TEMPLATE_PATH = "game_over_template.png" # Создайте этот файл!
GAME_OVER_DETECTION_THRESHOLD = 0.8
GAME_OVER_SEARCH_REGION = None # (x, y, w, h) относительно GAME_REGION

REWARD_ALIVE = 0.1
REWARD_DEATH = -10.0

# --- Сохранение модели ---
MODEL_SAVE_PATH = f"{PROJECT_NAME.lower()}_model.pth"
SAVE_MODEL_EVERY_N_EPISODES = 20
PLOT_SAVE_PATH = f"{PROJECT_NAME.lower()}_training_plots.png"

# --- Детектор иконки игрока (экспериментально) ---
PLAYER_ICON_TEMPLATE_PATH = "player_icon_template.png"
PLAYER_ICON_DETECTION_THRESHOLD = 0.7
PLAYER_EXPECTED_REGION = (10, FRAME_HEIGHT // 2 - 20, 80, 40)
PENALTY_PLAYER_NOT_FOUND = -0.5 # Небольшой штраф, если игрок не найден (может быть шумным)