    
import os

from cvnodes.util import CATEGORY_NAME, INPUT_DIR, get_annotated_filepath


class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = CATEGORY_NAME

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        print(f'srt_path:{srt}')
        srt_path = get_annotated_filepath(srt)
        return (srt_path,)
