import random
import time
import os
from playsound import playsound
import pickle
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utilityFunctions import featureExtraction

# SEANCE SIMULATOR CODE WITH CLASSIFIER
# This code runs, in one script, what SEANCE needs to do. This now has a classifier trained on GT data to run with.
# 1) Retrieve a number between 0-43 (corresponding to a phoneme or letter). This simulates the EEG BCI classifier output. 
# 2) Send that integer over AKLO. (Not performed here, but printed to command line for demo.)
# 3) Receive the integer in a message over AKLO. (NOT performed here.) 
# 4) Play the appropriate sound file (stored in StimPres folder). 
# 5) Reset after the refresh rate. (Here set to 100 ms, or 0.1 second.)
# NOTE on last step: I add '3' to the random number generator, simply because all sound file names run from 3-46, instead of 0-43. It is a simple offset. 

# Load classifier model (Bigger file!)
filename = 'gtMegaModel.sav'
#pickle.dump(clf, open(filename, 'wb'))
model = pickle.load(open(filename, 'rb'))


# Set fixed parameters.


# System parameters and feature extraction settings.

SAMPLE_RATE = 250  # Hz (subject to change)
fs=SAMPLE_RATE
NUM_WINDOWS = 2  # Dependent on number of samples of phonemes
lowcut=1
highcut=(np.floor(SAMPLE_RATE/2))
pcti = 99.95
activeChannels=int(16)

# How much delay between cycle iterations.
refreshrate=.1
time.sleep(refreshrate)

# Arbitrary offset value due to file names (Explained in NOTE). 
z=int(3)

# localize directory with sound files. 
absolute_path = os.path.dirname(__file__)

# Main loop
while z==int(3):


# Generate random data to simulate a new array of samples. This is the raw acquisition of data. 
	rawData=float(random.randint(0,43))*np.random.random((fs,activeChannels))

f1=rawData
	print(np.shape(rawData))

	f1a=featureExtraction(rawData[0,:], fs, lowcut, highcut, pcti)
f1b=featureExtraction(rawData[1,:], fs, lowcut, highcut, pcti)
f1c=featureExtraction(rawData[2,:], fs, lowcut, highcut, pcti)
f1d=featureExtraction(rawData[3,:], fs, lowcut, highcut, pcti)
f1e=featureExtraction(rawData[4,:], fs, lowcut, highcut, pcti)
f1f=featureExtraction(rawData[5,:], fs, lowcut, highcut, pcti)
f1f=featureExtraction(rawData[6,:], fs, lowcut, highcut, pcti)


f1=np.concatenate((f1a,f1b),axis=0)
	print(np.shape(data))

# Generate output integer. This simulates the EEG BCI classifier output. 
#	n=int(round(model.predict(data)))
	n = random.randint(0,43)
# Print integer. 
	print(n)
# Add offset (NOT required in final system). 
	updatedInteger=int(n+z)

# Generate file name from updated index. 
	audio_path = 'slide' + str(updatedInteger) + '.mp3'
	full_audio_path = os.path.join(absolute_path, audio_path)
# Play the appropriate sound.
	#playsound(full_audio_path)
# Delay for the next loop iteration. 
	time.sleep(refreshrate)
