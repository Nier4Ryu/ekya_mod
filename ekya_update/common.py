import sys, os, pandas as pd, ast, pytz, torch
from datetime import datetime

# Dino related...
DINOv3_MODEL_LOCAL_REPO = "../dinov3"  # Dino v3 git cloned dir
DINOv3_MODEL_WEIGHT_PATH_BASE = "/mnt/hdd_1/models/official_dino_v3/weights/lvd1689m_vit"  # Dino v3 weights path base dir

# Stopping sys
def stop_sys(message: str = None, raise_error: bool = False, temp_log: str = None):
    if not raise_error:
        try:
            # Console Log
            if message is not None:
                print(message)
            else:
                print("stopping sys, no message to show")

            sys.exit()
        except SystemExit as e:
            print(f"Normal Exit with sys.exit")
            raise  # re-raise to actually exit
        except Exception as e:
            print(f"Strange Error, {e}")
    else:
        if message is not None:
            raise RuntimeError(message)
        else:
            raise RuntimeError("no message given for this run time error raising")


# Getting KST
def get_kst():
    # Define KST timezone
    kst = pytz.timezone("Asia/Seoul")

    # Get the current time in KST
    current_time_kst = datetime.now(kst)

    return current_time_kst


def get_kst_as_string():
    kst = get_kst()
    kst_string = format_time_object_as_string(kst)
    return kst_string

def format_time_object_as_string(time_object):
    string_formated_time = time_object.strftime("%Y-%m-%d %H:%M:%S")
    return string_formated_time

# Generate Dirs
def generate_dirs(path):
    dir_path = os.path.dirname(path)
    if dir_path:  # only if non-empty
        os.makedirs(dir_path, exist_ok=True)


def atomic_torch_save(obj, path: str):
    tmp_path = path + ".tmp"
    try:
        # write checkpoint to temp file
        generate_dirs(path)
        with open(tmp_path, "wb") as f:
            torch.save(obj, f, pickle_protocol=5)
            f.flush()
            os.fsync(f.fileno())  # flush to disk
        # replace is atomic
        os.replace(tmp_path, path)
    finally:
        # if something failed, remove tmp
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# kwargs are used to pass arguments to ".to_csv" function -> ex: pass "index=False" option
def atomic_to_csv(df: pd.DataFrame, path: str, **kwargs):
    # Safe saving helper (strip index as default)
    if "index" not in kwargs:
        kwargs["index"] = False

    tmp_path = path + ".tmp"
    try:
        generate_dirs(path)
        with open(
            tmp_path, "w", newline="", encoding=kwargs.get("encoding", "utf-8")
        ) as f:
            df.to_csv(f, **kwargs)
            f.flush()
            os.fsync(f.fileno())  # ensure data is on disk
        os.replace(tmp_path, path)  # atomic rename
    finally:
        # if anything went wrong, clean up tmp
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def append_to_csv(append_item, path: str, **kwargs):
    # Safe saving helper (strip index as default)
    if "index" not in kwargs:
        kwargs["index"] = False

    # 1) Type Handle
    if isinstance(append_item, dict):
        df = pd.DataFrame(append_item)
    elif isinstance(append_item, pd.DataFrame):
        df = append_item
    else:
        message = f"type of {type(append_item)} is not defined!"
        stop_sys(message, raise_error=True)

    # 2) Save new or append
    if not os.path.exists(path):
        atomic_to_csv(df, path=path)
    else:
        # Safety checker for same cols
        existing_cols = pd.read_csv(path, nrows=0).columns.tolist()
        df = df.reindex(columns=existing_cols)

        # Add kwarg so that no headers are added!
        kwargs["header"] = False
        with open(path, "a", newline="", encoding=kwargs.get("encoding", "utf-8")) as f:
            df.to_csv(f, **kwargs)
            f.flush()
            os.fsync(f.fileno())


# Applying ast.literal.eval on col
def apply_ast_on_col(df, col):
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df
