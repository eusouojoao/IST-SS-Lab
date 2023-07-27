# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:17:18 2015

@author: yoda
"""

try:
    # Attempt to load winsound
    from winsound import PlaySound
except ImportError:
    import subprocess
    def PlaySound(filename, nonblocking):
        # Try afplay command (Mac OS X)
        ret = subprocess.call("afplay %s %s"%(filename, ('&' if nonblocking else '')),
                              shell=True, stderr=subprocess.DEVNULL)
        if ret==0:
            return
        # Try aplay command (Linux)
        ret = subprocess.call("aplay %s %s"%(filename, ('&' if nonblocking else '')),
                              shell=True, stderr=subprocess.DEVNULL)
        if ret==0:
            return
        # Unable to play sound
        print("Unable to play sound on this platform")

