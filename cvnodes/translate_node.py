import logging
import os

from cvnodes.subtitle import process_mergin, load_srt, remove_html_tag, save_subtitles_to_srt, save_subtitles_to_text
from cvnodes.translate_by_gemini import set_proxy, translate_by_multiline
from cvnodes.util import CATEGORY_NAME, OUTPUT_DIR

logger = logging.getLogger(__name__)

class TranslateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_file": ("SRT",),  # 接受 SRT 文件作为输入
                "max_lines":("INT",{
                    "default":10
                })
            }
        }

    RETURN_TYPES = ("SRT","SRT")  # 输出 SRT 文件路径和文本内容
    RETURN_NAMES = ("SRT","SRT_Text") 
    FUNCTION = "process"
    CATEGORY = CATEGORY_NAME

    def process(self, srt_file: str,max_lines=10):
        """
        翻译字幕
        """
        if not os.path.isfile(srt_file):
            raise ValueError(f"SRT file not found: {srt_file}")
        
        #翻译后的输出文件
        output_translate = os.path.join(OUTPUT_DIR,"output-translated.srt")
        output_text_cn = os.path.join(OUTPUT_DIR,"output-中文文本.txt")

        subtitles = load_srt(srt_file)

        print('设置本地代理')
        set_proxy()
        
        print(f'读取字幕文件：{srt_file}')
        subtitles = load_srt(srt_file)

        print(f'开始翻译:({len(subtitles)})')
        new_subtitles = translate_by_multiline(subtitles,output_translate)
        save_subtitles_to_text(new_subtitles,output_text_cn)
        return (output_translate, output_text_cn) # 返回原始文件路径和解析后的文本内容

