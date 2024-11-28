import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvnodes.load_srt_node import LoadSRT
from cvnodes.subtitle_node import SrtMergeNode
from cvnodes.cosyvoice_node import  CosyVoiceNode, CosyVoiceDubbingNode, CosyVoiceSrtNode
from cvnodes.text_node import TextNode, TextView
from cvnodes.translate_node import TranslateNode


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LoadSRT":LoadSRT,
    "TextNode": TextNode,
    "CosyVoiceNode": CosyVoiceNode,
    "CosyVoiceDubbingNode":CosyVoiceDubbingNode,
    "SrtMergeNode":SrtMergeNode,
    "TranslateNode":TranslateNode,
    "TextView":TextView,
    "CosyVoiceSrtNode":CosyVoiceSrtNode
}
