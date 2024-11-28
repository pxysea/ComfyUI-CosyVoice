from pathlib import Path
import os,sys

def find_project_root(current_path: Path, marker: str = ".git") -> Path:
    """
    Recursively find the project root based on a marker file or directory.
    :param current_path: The starting path to search.
    :param marker: Marker file or directory indicating the root (default is '.git').
    :return: The root directory as a Path object.
    example: # Start searching from the current script's directory
            current_path = Path(__file__).resolve()
            project_root = find_project_root(current_path)
    """
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root not found. Marker '{marker}' is missing.")

ROOT_DIR = find_project_root(Path(__file__).resolve(),"requirements.txt")
print(f'root_dir:{ROOT_DIR}')
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'third_part'))

# ComfyUI config
import folder_paths
CATEGORY_NAME = "ComfyUI-CosyVoice"
INPUT_DIR = folder_paths.get_input_directory()
OUTPUT_DIR = os.path.join(folder_paths.get_output_directory(),"cosyvoice_dubb")
pretrained_models = os.path.join(ROOT_DIR,"pretrained_models")

def get_annotated_filepath(srt_file):
    return folder_paths.get_annotated_filepath(srt_file)