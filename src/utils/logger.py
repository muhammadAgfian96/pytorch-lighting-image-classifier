import logging
from rich import print

class LoggerStdOut:
    def title_section(self, *args):
        len_char = len(*args)+9
        print('▪️'*len_char)
        print('🛠️ ', *args, ' 🛠️')
        print('_'*len_char)
    
    def sub_section(self, *args):
        print("🔰 sub:", *args, " 🔰")