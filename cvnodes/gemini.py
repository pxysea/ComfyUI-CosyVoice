
import google.generativeai as genai
import os

class Gemini:
    def __init__(self,api_key = None,max_tokens=8192) -> None:
        if not api_key:
            self.api_key = os.environ["GOOGLE_API_KEY"]
        else:
            self.api_key = api_key
        # max_output_tokens
        self.max_tokens=max_tokens
        # self.model_name="gemini-1.0-pro"
        # self.model_name="gemini-1.5-pro"
        self.model_name="gemini-1.5-flash"

        self.init()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name,safety_settings = self.safety_settings,generation_config = self.generation_config)
        # self.model = genai.GenerativeModel('gemini-pro',self.safety_settings,self.generation_config)
        pass

    def init(self):
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        # self.generation_config = {
        #     "temperature":0,
        #     "top_p":1,
        #     "top_k":1,
        #     "max_output_tokens":self.max_tokens
        # }
        self.generation_config = {
            "temperature":1,
            "top_p":0.95,
            "top_k":30,
            "max_output_tokens":self.max_tokens
        }

    def translate(self,content):
        '''
            翻译内容
        '''
        text = f'翻译为简体中文: \n{content}'
        response = self.model.generate_content(text)
        reply_text = response.text
        # print(reply_text)
        return reply_text
    
    def generate(self,content):
        '''
            内容
        '''
        # text = f'翻译为简体中文: \n{content}'
        response = self.model.generate_content(content)
        reply_text = response.text
        # print(reply_text)
        return reply_text
# def main():
    
#     gemini = Gemini()
#     content = gemini.translate('The tech world is about to witness one of the most anticipated breakthroughs.')
#     print(content)
    

# if __name__ == "__main__":
#     main()
