from pathlib import Path

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

    
