
import logging
import os

from cvnodes.subtitle import process_mergin, load_srt, remove_html_tag, save_subtitles_to_srt
from cvnodes.util import CATEGORY_NAME, OUTPUT_DIR


logger = logging.getLogger(__name__)

class SrtMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_file": ("SRT",),  # 接受 SRT 文件作为输入
                "maxchars":("INT",{
                    "default":90
                })
            }
        }

    RETURN_TYPES = ("SRT","TEXT")  # 输出 SRT 文件路径和文本内容
    FUNCTION = "process_merge_srt"
    CATEGORY = CATEGORY_NAME

    def process_merge_srt(self, srt_file: str,maxchars=90):
        """
        解析 SRT 文件，将其内容提取为纯文本。
        """
        if not os.path.isfile(srt_file):
            raise ValueError(f"SRT file not found: {srt_file}")
        
        subtitles = load_srt(srt_file)

        subtitles = remove_html_tag(subtitles)

        processed_subtitles = process_mergin(subtitles,max_chars=maxchars)
        output_file = os.path.join(OUTPUT_DIR,"merged_srt.srt")
        save_subtitles_to_srt(processed_subtitles, output_file)

        full_text = ' '.join(entry.text for entry in subtitles)
        logger.info(f'full text:{full_text}')
        return (output_file, full_text) # 返回原始文件路径和解析后的文本内容



# 注册节点
NODE_CLASS_MAPPINGS = {
    "SrtMergeNode": SrtMergeNode
}

# 可选：设置节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "SrtMergeNode": "SRT Merge Node"
}