"""
    Import libraries
"""
import sys
from engine import *

"""
    Console outputs and inputs
"""
print("Welcome to your CNN Classifier! Which option do you want to perform?")
print("1. Check a 'Layer vs. Accuracy' Analysis")
print("2. Check a 'Layer vs. Accuracy' using CNNs and dropout layers")
print("3. Check how one of our trained CNNs performs")
selected_option = input('--> ')
preload__data()

if selected_option == '1': 
    layer__vs__accuracy(False)    
elif selected_option == '2':
    layer__vs__accuracy(True)
elif selected_option == '3':
    print("Pick one our trained CNNs and see how it predicts (more info about them in the README.md):")
    print("1. Basic CNN")
    print("2. Deeper dilated CNN")
    print("3. Deeper CNN Classificator with dropouts")
    print("4. Deeper CNN Classificator with batchnorm (this one is our favorite!)")
    selected_cnn = input('--> ')
    load__and__predict(selected_cnn)
else:
    print('Thanks for coming along!')