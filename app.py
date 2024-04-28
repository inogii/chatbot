import gradio as gr
import src.pdfparser.neuralsearcher as pp
import pygetwindow as gw
import pyautogui
import subprocess
import os
import imageio as iio

searcher = pp.NeuralSearcher('student_manual')


def take_screenshot_mac(prompt, limit):
    window_title = "Google Chrome"
    path = '/tmp/window_title_screenshot.png'
    script = f'''
    tell application "System Events"
        set frontmost of the first process whose name is "{window_title}" to true
        delay 1
    end tell
    do shell script "screencapture -W {path}"
    '''
    subprocess.run(["osascript", "-e", script], check=True)

    #screenshot = iio.imread('/tmp/window_title_screenshot.png')

    return searcher.image_search(path, prompt=prompt, limit=limit, verbose=(2,1))


def take_screenshot():
    # Get a list of all windows
    all_windows = gw.getAllWindows()
    # Filter for Chrome windows
    chrome_windows = [win for win in all_windows if 'Chrome' in win.title]
    
    if chrome_windows:
        # Assuming you want to screenshot the first Chrome window found
        chrome_window = chrome_windows[0]
        # Bring the window to the foreground (this may be required on some systems)
        chrome_window.activate()
        # Make sure the window is not minimized
        chrome_window.restore()
        # Use pyautogui to take a screenshot of the specified area
        screenshot = pyautogui.screenshot(region=(
            chrome_window.left, 
            chrome_window.top, 
            chrome_window.width, 
            chrome_window.height
        ))
        # Save the screenshot
        screenshot_path = 'screenshot.png'
        screenshot.save(screenshot_path)

        return searcher.image_search(screenshot, verbose=(2,0))

    else:
        print("No Chrome window found.")


def create_interface():
    with gr.Blocks() as demo:
        # Input section
        with gr.Row():
            prompt_input = gr.Textbox(label="Prompt")
        with gr.Row():
            limit_input = gr.Slider(minimum=1, maximum=10, step=1, label="Limit")
        
        # Button section
        with gr.Row():
            screenshot_button = gr.Button("Take Screenshot")
        
        # Output section
        with gr.Row():
            output = gr.Textbox(label="Output")
        
        screenshot_button.click(
            take_screenshot_mac, 
            [prompt_input, limit_input], 
            output
        )
    
    demo.launch()

create_interface()


