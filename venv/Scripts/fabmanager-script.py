#!C:\Users\HWer\PycharmProjects\InsilicoTestProject\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'flask-appbuilder==1.13.0','console_scripts','fabmanager'
__requires__ = 'flask-appbuilder==1.13.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('flask-appbuilder==1.13.0', 'console_scripts', 'fabmanager')()
    )
