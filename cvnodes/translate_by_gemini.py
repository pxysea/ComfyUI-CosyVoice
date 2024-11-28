
import os
import re
import sys
import time

from .subtitle import load_srt, parse_srt_to_objects, save_subtitles_to_srt
# sys.path.append(os.getcwd())
from .gemini import Gemini
from .progress import *

gemai = Gemini()

def set_proxy(proxy='http://localhost:1081'):
    os.environ['http_proxy']=proxy
    os.environ['https_proxy']=proxy

def gen_srt_text(start_index,end_index,subtitles):
    '''
    生成字幕文本
    '''
    srt_text = ''
    # selected_subtitles = [sub for sub in subtitles if start_index <= sub.index <= end_index]    
    
    i = start_index
    if(end_index>len(subtitles)):
        end_index = len(subtitles)
    for i in range(start_index,end_index+1):
        st = subtitles[i-1]
        srt_text += f"{st.index}\n{st.start_time_str} --> {st.end_time_str}\n{st.text}\n\n"

    return srt_text

def process_singleline(subtitles,cache_st = None):
    '''
    翻译以单行模式
    '''
    new_subtitles = []
    start_index = 0
    if cache_st:
        new_subtitles = cache_st[:]
        start_index = len(new_subtitles)
        print(f'start index:{start_index}')
    try :
        for i,entry in enumerate(subtitles):
            if i>=start_index:
                e1 = entry.copy()
                text = entry['text']
                resp = gemai.translate(content=text)
                e1['text'] = resp
                new_subtitles.append(e1)
                print(f'{e1}')
                time.sleep(1)
    except :
        print('出现异常')
    return new_subtitles

def translate(text=''):
    pre_text = "把下面英文字幕文本翻译为简体中文，要求：1、并保留原字幕格式输出， 2、翻译侧重使用航天专业术语\n"
    content = f"{pre_text}{text}"
    txt = ''
    try:
        txt = gemai.generate(content=content)
    except:
        txt = ''
        
    return txt

def process_multiline(subtitles):
    '''
    翻译字幕信息以多行方式
    '''
    pre_text = "SRT 字幕，以 srt 格式输出：\n"
    per_count = 10
    i = 1
    processed = []

    print_progress_bar(0,len(subtitles),prefix="翻译进度:",suffix="完成",length=50)
    while i<len(subtitles):
        end = i+per_count-1
        if end > len(subtitles): end = len(subtitles)
        
        srt_text = gen_srt_text(i,end,subtitles)
        txt = ''
        
        txt = translate(srt_text)
        ## 重试
        if txt == '':
            time.sleep(2)
            try:
                txt = translate(srt_text)            
            except:
                txt = ''
                print(f'重试失败:{i}')

        #替换返回 
        txt = re.sub(re.compile(f'^{pre_text}\n'),'',txt)
        txt = re.sub(r'^```srt\n','', txt)
        txt = re.sub(r'```','', txt)

        tmp = parse_srt_to_objects(txt)
        #修正序号
        if len(tmp)>0 and tmp[0].index != i:
             for idx,item in enumerate(tmp):
                tmp[idx].index=i+idx
        if(len(tmp) != (end-i+1)):
            print(f'返回数量不同:{i}')
        processed += tmp
        i+= per_count
        print_progress_bar(end,len(subtitles),prefix="翻译进度:",suffix="完成",length=50)
        time.sleep(2)

    return processed


def translate_by_singleline(subtitles,output_file):
    '''
    按照字幕序号，一次翻译一个字幕
    '''
    cache_st = None
    if os.path.exists(output_file):
        cache_st = load_srt(output_file)

    new_subtitles = process_singleline(subtitles,cache_st)
    print('写入文件')
    save_subtitles_to_srt(new_subtitles,output_file)
    print(f'任务完成!')

def translate_by_multiline(subtitles,output_file):
    '''
    按照字幕序号，一次翻译一个字幕
    '''
    new_subtitles = process_multiline(subtitles)
    print(f'处理翻译完成，保存文本: {output_file}')

    save_subtitles_to_srt(new_subtitles,output_file)
    print(f'任务完成 {len(subtitles)} -> {len(new_subtitles)}')
    return new_subtitles

def parse_to_translation_train_json(en_subtitles,cn_subtitles):
    '''
    生成翻译训练数据
    '''
    i = 1
    processed = []
    for st in en_subtitles:
        text = f"{st['index']}\n{st['start']} --> {st['end']}\n{st['text']}\n\n"
        # print(text);

    while i<len(en_subtitles):
        s_text = en_subtitles[i]['text']        
        t_text = cn_subtitles[i]['text']
        processed += {"source":s_text,"target":t_text,"transllation":t_text}


    return processed

