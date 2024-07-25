import pyttsx3

text_speech=pyttsx3.init()

answer=input("What would you like to listen:")

text_speech.say(answer)
text_speech.runAndWait()