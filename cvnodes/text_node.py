
from cvnodes.util import CATEGORY_NAME

class TextNode:
    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})},
        }
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "Text"

    def encode(self,text):
        return (text ,)
    
class TextView():
    
    RETURN_TYPES = ()
    FUNCTION = "encode"

    CATEGORY = "Text"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):        
        return {
            "required": {
                "text": ("TEXT", 
                         {
                            "multiline": True, 
                            "dynamicPrompts": True,
                            # "lazy": True,
                            "default":"Test"
                         })
            },
        }
    
    def encode(self,text):
        print(f'text:{text}')
        return(text)