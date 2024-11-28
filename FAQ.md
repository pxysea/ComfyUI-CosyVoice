### 后台异常 "ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接"
异常信息如下：
```
Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
handle: <Handle _ProactorBasePipeTransport._call_connection_lost(None)>
Traceback (most recent call last):
  File "D:\Python\python312\Lib\asyncio\events.py", line 88, in _run
    self._context.run(self._callback, *self._args)
  File "D:\Python\python312\Lib\asyncio\proactor_events.py", line 165, in _call_connection_lost
    self._sock.shutdown(socket.SHUT_RDWR)
ConnectionResetError: [WinError 10054] 远程主机强迫关闭了一个现有的连接。
```
**修复方法：**
修改 Python目录下的："Python\Lib\asyncio\windows_events.py" 文件 
最后一行：
```python
#DefaultEventLoopPolicy = WindowsProactorEventLoopPolicy
DefaultEventLoopPolicy = WindowsSelectorEventLoopPolicy
```
