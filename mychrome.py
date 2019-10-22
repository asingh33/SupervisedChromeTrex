"""
Created on Thu Oct  17 01:01:43 2019

@author: abhisheksingh
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

#This function will do initial webdriver setup for ChromeBrowser
def setup():
    chrome_options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
    driver.set_window_size(800, 600)
    #Open Dino game page in Chrome
    driver.get('chrome://dino/')
    return driver
    
#This function will send the SPACE key to make dino jump
def dinojump(browser):
    browser.send_keys(Keys.SPACE)

