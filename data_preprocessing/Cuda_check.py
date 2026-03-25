SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN = device.type == "cuda"
print(f"Using device: {device}")



#DATASET LOADING
DATA_DIR = "/kaggle/input/cattle-breeds-dataset/Cattle Breeds"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)