# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:13:37 2019

@author: gurkan.sahin
"""

class Utility(object):
    def __init__(self):
        pass
     

    @staticmethod
    def elapsed_time(start_t, end_t):
        e = int(end_t - start_t)
        hours = e // 3600
        e -= 3600 * hours
        minutes = e // 60
        seconds = e - 60 * minutes
        return str(hours) + ":" + str(minutes) + ":" + str(seconds)



