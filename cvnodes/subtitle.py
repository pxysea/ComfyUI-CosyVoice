from datetime import timedelta
import json
import os
import re


class Subtitle:
    
    def __init__(self, index:int, start:timedelta, end:timedelta, text:str,language="en"):
        """
        初始化字幕对象
        :param index: 序号
        :param start: 开始时间
        :param end: 结束时间
        :param text: 字幕文本
        """
        self.index = index
        self.start = start
        self.end = end
        self.text = text
        self.language = language  # 新增语言属性

    @classmethod
    def from_str_time(cls, index:int, start:str, end:str, text:str,language="en"):
        
        """
        初始化字幕对象
        :param index: 序号
        :param start: 开始时间字符串
        :param end: 结束时间字符串
        :param text: 字幕文本
        """
        return cls(index,parse_time(start),parse_time(end),text,language)

        

    def filter_by_language(subtitles, target_language):
        """
        按语言过滤字幕。
        :param subtitles: Subtitle 对象列表
        :param target_language: 目标语言（如 'en', 'zh'）
        :return: 过滤后的 Subtitle 对象列表
        """
        return [subtitle for subtitle in subtitles if subtitle.language == target_language]
    
    def _parse_time(self, time_str):
        """将时间字符串解析为 timedelta 对象"""
        hours, minutes, seconds = map(float, time_str.replace(",", ".").split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    def format_time(self, time_obj):
        """将 timedelta 对象格式化为 SRT 标准时间字符串"""
        total_seconds = int(time_obj.total_seconds())
        milliseconds = int(time_obj.microseconds / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    
    @property
    def start_time_str(self):
        """获取格式化的开始时间字符串"""
        return self.format_time(self.start)

    @property
    def end_time_str(self):
        """获取格式化的结束时间字符串"""
        return self.format_time(self.end)
    
    @property
    def total_seconds(self):        
        return (self.end - self.start).total_seconds()

    
    @property
    def length(self):        
        return len(self.text)
    
    def adjust_time(self, milliseconds):
        """调整开始和结束时间"""
        delta = timedelta(milliseconds=milliseconds)
        self.start = max(self.start + delta, timedelta(0))  # 确保时间非负
        self.end = max(self.end + delta, timedelta(0))

    def to_dict(self):
        """导出为字典"""
        return {
            "index": self.index,
            "start": self.start_time_str,
            "end": self.end_time_str,
            "text": self.text
        }
    
    def __repr__(self):
        """返回对象的字符串表示"""
        return f"Subtitle(index={self.index}, start='{self.start}', end='{self.end}', text='{self.text}')"
    
def parse_time( time_str):
        """将时间字符串解析为 timedelta 对象"""
        hours, minutes, seconds = map(float, time_str.replace(",", ".").split(":"))
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def ms_to_srt_time(ms):
    """
    将毫秒时间转换为 SRT 格式时间 (hh:mm:ss,SSS)
    """
    N = int(ms)
    hours, remainder = divmod(N, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_srt_to_objects(srt_content):
    """
    解析 SRT 文件并返回 Subtitle 对象的列表。
    :param srt_content: SRT 文件内容
    :return: Subtitle 对象列表
    """
    lines = [line.strip() for line in srt_content.strip().split('\n') if line.strip()]
    subtitles = []
    i = 0
    
    while i < len(lines):
        if lines[i].isdigit():
            # 读取序号
            index = int(lines[i])
            i += 1
            
            # 读取时间码
            times = lines[i]
            start, end = times.split(' --> ')
            i += 1
            
            # 读取字幕文本
            text = ""
            while i < len(lines) and lines[i] and not lines[i].isdigit():
                text += lines[i] + '\n'
                i += 1
            text = text.strip()

            # 创建 Subtitle 对象
            subtitle = Subtitle.from_str_time(index=index, start=start, end=end, text=text)
            subtitles.append(subtitle)
        else:
            i += 1
    return subtitles

def load_srt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return parse_srt_to_objects(f.read())
    
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 将所有字幕导出为 JSON
def export_subtitles_to_json(subtitles, output_path):
    """
    导出字幕为 JSON 文件
    :param subtitles: Subtitle 对象列表
    :param output_path: 输出 JSON 文件路径
    :example: # 示例导出
    :export_subtitles_to_json(subtitles, "subtitles.json")
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([subtitle.to_dict() for subtitle in subtitles], f, indent=4, ensure_ascii=False)

# 将字幕保存为 SRT 文件
def save_subtitles_to_srt(subtitles, output_path):
    """
    保存字幕为 SRT 文件
    :param subtitles: Subtitle 对象列表
    :param output_path: 输出 SRT 文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for subtitle in subtitles:
            f.write(f"{subtitle.index}\n")
            f.write(f"{subtitle.start_time_str} --> {subtitle.end_time_str}\n")
            f.write(f"{subtitle.text}\n\n")

# 将字幕保存为 SRT 文件
def save_subtitles_to_text(subtitles, output_path):
    """
    保存字幕为 SRT 文件
    :param subtitles: Subtitle 对象列表
    :param output_path: 输出 text 文件路径
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for subtitle in subtitles:
            f.write(f"{subtitle.text}\n")

def batch_process_srt(input_folder, output_folder, time_adjustment=0):
    """
    批量处理 SRT 文件：调整时间并保存。
    :param input_folder: 输入 SRT 文件所在目录
    :param output_folder: 输出文件存放目录
    :param time_adjustment: 时间调整量（毫秒）
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".srt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取并解析 SRT
            with open(input_path, "r", encoding="utf-8") as f:
                srt_content = f.read()
            subtitles = parse_srt_to_objects(srt_content)
            
            # 调整时间
            for subtitle in subtitles:
                subtitle.adjust_time(time_adjustment)
            
            # 保存调整后的字幕
            save_subtitles_to_srt(subtitles, output_path)
            print(f"Processed: {filename}")

def merge_srt_files(file_list, output_path):    
    """
    合并多个 SRT 文件。
    :param file_list: 字幕文件路径列表
    :param output_path: 输出文件路径
    :Example: merge_srt_files(["file1.srt", "file2.srt"], "merged_output.srt")
    """
    merged_subtitles = []
    current_index = 1

    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            subtitles = parse_srt_to_objects(f.read())
        
        for subtitle in subtitles:
            subtitle.index = current_index
            current_index += 1
            merged_subtitles.append(subtitle)
    
    save_subtitles_to_srt(merged_subtitles, output_path)
    print(f"Successfully merged into: {output_path}")            


def adjust_time(original_time:timedelta, milliseconds: int) -> timedelta:
    """
    在 SRT 时间字符串上添加或减少指定毫秒数。
    """
    # original_time = parse_srt_time(srt_time)
    adjusted_time = original_time + timedelta(milliseconds=milliseconds)
    if adjusted_time.total_seconds() < 0:
        adjusted_time = timedelta(0)
    return adjusted_time

def remove_html_tag(subtitles):
    for st in subtitles:
        st.text = re.sub(r'<[^>]+>', '', st.text)
    return subtitles


def process_mergin(entries, max_chars=90, min_chars=30):

    '''
        处理英文字幕分割/合并，方式：以标点符号进行分割 合并处理
    '''
    processed = []
    buffer_text = ""
    buffer_start_td = None
    buffer_end_td = None
    
    for entry in entries:
        
        time_rate = (entry.total_seconds*1000) / entry.length

        # 简单地根据[,.!?]拆分句子
        sentences = re.split(r'((?<!\d)[,.!?，。！？](?=\s|$)|(?<!\d),(?=\s))', entry.text + ' ')
        sentences = [item for item in sentences if item.strip()]  # 移除空字符
        
        if not buffer_text:
            buffer_start_td = entry.start
        
        if len(sentences) > 1:
            temp_str = ''
            for j in range(0, len(sentences), 2):
                temp_str = ''.join(sentences[j:j+2])
                # 匹配 ?!. 切断
                if j < len(sentences) - 1 and (re.search(r'(?<!\d)[?.!？。！](?!\d)|\?', sentences[j+1]) 
                        or 
                        (sentences[j+1] == ',' and len(''.join([buffer_text,temp_str]))>max_chars)):
                    
                    buffer_text += ' ' + temp_str 
                    if j == 0:
                        buffer_end_td = adjust_time(entry.start, (1 + len(temp_str)) * time_rate)
                    elif j < len(sentences) - 2:
                        
                        if len(processed)>0:                        
                            buffer_end_td = adjust_time(processed[-1].end , (1 + len(temp_str)) * time_rate)
                        else:
                            buffer_end_td = adjust_time(entry.start , (1 + len(temp_str)) * time_rate)
                        
                    else:
                        buffer_end_dt = entry.end

                    processed.append(Subtitle(index=len(processed) + 1, start=buffer_start_td, end=buffer_end_td, text=buffer_text.strip()))
                    buffer_text = ''
                    temp_str = ''
                    buffer_start_td = buffer_end_td

                buffer_text += temp_str
        elif re.search(r'[\]]$', entry.text):
            #特殊处理 如: [Music]
            processed.append(Subtitle(index=len(processed) + 1, start=entry.start, end=entry.end, text=entry.text))
            buffer_text = ''
        else:
            if not buffer_text:
                buffer_start_td = entry.start
            else:
                buffer_text += " "
            buffer_text += entry.text

        if len(buffer_text)>max_chars:  # 没有标点时处理
            processed.append(Subtitle(index=len(processed) + 1, start=buffer_start_td, end=entry.end, text=buffer_text))
            buffer_text = ''

    if buffer_text: # 最后条
        processed.append(Subtitle(index=len(processed) + 1, start=buffer_start_td, end=entry.end, text=buffer_text))

    return processed

