import torch

# ⚙️ Общие настройки
PROJECT_NAME = "GDint_vUltra"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_TITLE_SUBSTRING = "Geometry Dash"

# 🔄 Управление и отладка
PAUSE_RESUME_KEY = 'p'
LOG_LEVEL = "INFO"
LOG_FILE = f"{PROJECT_NAME.lower()}_log.txt"
SAVE_FRAMES_ON_DEATH = True
SHOW_GAME_REGION_OUTLINE = True
GAME_REGION_OUTLINE_BORDER_COLOR = "lime"
GAME_REGION_OUTLINE_THICKNESS = 2
DYNAMIC_WINDOW_TRACKING_INTERVAL = 0.25 

# 🖼️ GUI
ENABLE_GUI = True
GUI_UPDATE_INTERVAL_MS = 20
GUI_SHOW_RAW_CAPTURE = True
GUI_RAW_CAPTURE_DISPLAY_SCALE = 0.6
GUI_AI_VIEW_DISPLAY_SCALE = 4
GUI_MARK_DETECTED_OBJECTS = True

# 📸 Захват
FALLBACK_GAME_REGION = None
FRAME_WIDTH = 96
FRAME_HEIGHT = 96
GRAYSCALE = True
NUM_FRAMES_STACKED = 4

# 🎯 AI параметры (на "ультрах")
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRAMES = 20000
BATCH_SIZE = 128
REPLAY_MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ_EPISODES = 2
LEARN_START_STEPS = 1500
MAX_STEPS_PER_EPISODE = 1000
NUM_EPISODES = 20000
NUM_ACTIONS = 2
ACTION_DELAY = 0.01
JUMP_DURATION = 0.07
AI_FPS_LIMIT = 240

# 🧠 Поддержка обучения
MODEL_SAVE_PATH = f"{PROJECT_NAME.lower()}_model.pth"
SAVE_MODEL_EVERY_N_EPISODES = 10
PLOT_SAVE_PATH = f"{PROJECT_NAME.lower()}_training_plots.png"

# 🎮 Обнаружение "смерти" и объектов
GAME_OVER_SCREEN_TEMPLATE_PATH = "game_over_template.png"
GAME_OVER_SCREEN_DETECTION_THRESHOLD = 0.7
PLAYER_DEATH_EFFECT_TEMPLATE_PATH = "player_shatter_template.png"
PLAYER_DEATH_EFFECT_THRESHOLD = 0.65
SPIKE_TEMPLATE_PATH = "spike_template.png"
SPIKE_DETECTION_THRESHOLD = 0.60
MAX_SPIKES_TO_DRAW = 5
GAME_OVER_SEARCH_REGION = None

# 🏆 Вознаграждения
REWARD_ALIVE = 0.01
REWARD_DEATH = -1.0
REWARD_PROGRESS_FACTOR = 0.8

# 🚀 Дополнительно (для продвинутого пользователя)
USE_MIXED_PRECISION = True
ENABLE_PROFILING = False
torch.backends.cudnn.benchmark = True
