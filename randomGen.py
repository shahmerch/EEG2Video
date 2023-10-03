import random
import time
import os
from playsound import playsound

# SEANCE SIMULATOR CODE
# This code runs, in one script, what SEANCE needs to do.
# 1) Retrieve a number between 0-43 (corresponding to a phoneme or letter). This simulates the EEG BCI classifier output.
# 2) Send that integer over AKLO. (Not performed here, but printed to command line for demo.)
# 3) Receive the integer in a message over AKLO. (NOT performed here.) 
# 4) Play the appropriate sound file (stored in StimPres folder). 
# 5) Reset after the refresh rate. (Here set to 100 ms, or 0.1 second.)
# NOTE on last step: I add '3' to the random number generator, simply because all sound file names run from 3-46, instead of 0-43. It is a simple offset. 


# Set fixed parameters.

# How much delay between cycle iterations.
refreshrate=.1
time.sleep(refreshrate)

# Arbitrary offset value due to file names (Explained in NOTE). 
z=int(3)

# localize directory with sound files. 
absolute_path = os.path.dirname(__file__)

# Main loop
while z==int(3):
# Generate random integer. This simulates the EEG BCI classifier output. 
	n = random.randint(0,43)
# Print integer. 
	print(n)
# Add offset (NOT required in final system). 
	updatedInteger=int(n+z)

# Generate file name from updated index. 
	audio_path = 'slide' + str(updatedInteger) + '.mp3'
	full_audio_path = os.path.join(absolute_path, audio_path)
# Play the appropriate sound.
	playsound(full_audio_path)
# Delay for the next loop iteration. 
	time.sleep(refreshrate)
