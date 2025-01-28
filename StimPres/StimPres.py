import os
import pyautogui
import time
import random
import threading
import keyboard

# Create an Event object to manage pausing
pause_event = threading.Event()
pause_event.set()  # Initially, allow execution

# Function to toggle the pause state
def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()  # Pause execution
        print("Paused")
    else:
        pause_event.set()  # Resume execution
        print("Resumed")

# Start a thread to listen for the "x" key to toggle pause/resume
def listen_for_pause():
    keyboard.on_press_key("x", lambda _: toggle_pause())
    keyboard.wait()  # Keep this thread alive to listen for keypresses

# Slides and test setup
slides = [
    ['boat_1', 2], ['boat_2', 2], ['dog_1', 2], ['dog_2', 2],
    ['apple_1', 2], ['apple_2', 2], ['cat_1', 2], ['cat_2', 2],
    ['bowling_1', 2], ['bowling_2', 2], ['banana_1', 2], ['banana_2', 2],
]

test_slides = [
    ['boat_1', 0], ['boat_2', 0], ['dog_1', 0], ['dog_2', 0],
    ['apple_1', 0], ['apple_2', 0], ['cat_1', 0], ['cat_2', 0],
    ['bowling_1', 0], ['bowling_2', 0], ['banana_1', 0], ['banana_2', 0],
]

absolute_path = os.path.dirname(__file__)
relative_path = "stimulus.pptx"
full_path = os.path.join(absolute_path, relative_path)

num_tests = 24
wait_slide = '15'
begin_test_slide = '2'
blank_slide = '16'

# Open the PowerPoint file in fullscreen
os.system("open " + full_path)
time.sleep(2)
pyautogui.hotkey(full_path, 'f5')
time.sleep(2)

# Rules slide
pyautogui.press('enter')
time.sleep(10)  # Allow time to read the rules

# Function to present a single block of tests
def one_block(slides_place):
    pause_event.wait()  # Wait if paused
    current_slide = slides_place + 3
    str_current_slide = str(current_slide)

    # Go to the slide
    pyautogui.press(str_current_slide[0])
    if current_slide >= 10:
        pyautogui.press(str_current_slide[1])
    pyautogui.press('enter')

    time.sleep(5)  # Time to analyze the picture

    for i in range(1, 11):
        pause_event.wait()  # Wait if paused
        test_slides[slides_place][1] += 1

        # Load wait slide
        pyautogui.press(wait_slide[0])
        pyautogui.press(wait_slide[1])
        pyautogui.press('enter')
        time.sleep(1)

        # Load blank slide
        pyautogui.press(blank_slide[0])
        pyautogui.press(blank_slide[1])
        pyautogui.press('enter')
        time.sleep(2)

    # Final wait slide
    pyautogui.press(wait_slide[0])
    pyautogui.press(wait_slide[1])
    pyautogui.press('enter')
    time.sleep(2)

# Main loop
def main_loop():
    global num_tests

    try:
        while num_tests > 0:
            pause_event.wait()  # Wait if paused

            if num_tests > 12:
                slides_place = random.randint(0, 5)
                if slides[slides_place][1] > 0:
                    slides[slides_place][1] -= 2
                    num_tests -= 2
                    one_block(slides_place)
            else:
                slides_place = random.randint(6, 11)
                if slides[slides_place][1] > 0:
                    slides[slides_place][1] -= 2
                    num_tests -= 2
                    one_block(slides_place)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        pyautogui.hotkey('esc')  # Exit fullscreen

    # Print results
    for i in range(12):
        print(test_slides[i][1])

# Launch the program
if __name__ == "__main__":
    # Start the listener for pause/resume
    threading.Thread(target=listen_for_pause, daemon=True).start()

    # Run the main loop
    main_loop()
