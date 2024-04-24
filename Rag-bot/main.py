from ultralytics import YOLO
from threading import Thread
import pydirectinput
import threading
import time
import pyautogui


screen_center_x = 1920 / 2
screen_center_y = 1080 / 2
area_radius = 500
lock = threading.Lock()


def autosell():
    time.sleep(1)
    pydirectinput.press('f4')
    time.sleep(0.2)
    pyautogui.press('f8')
    time.sleep(0.5)
    pyautogui.moveTo(x=1035, y=799)
    time.sleep(0.5)
    pydirectinput.keyDown('f5')
    time.sleep(0.2)
    pydirectinput.keyUp('f5')
    pyautogui.moveTo(x=857, y=534)
    time.sleep(0.5)
    pydirectinput.keyDown('f5')
    time.sleep(2)
    pydirectinput.keyUp('f5')
    time.sleep(0.5)
    pyautogui.moveTo(x=672, y=588)
    time.sleep(0.5)
    pydirectinput.keyDown('f5')
    time.sleep(0.2)
    pydirectinput.keyUp('f5')
    pydirectinput.press('f4')
    time.sleep(0.2)
    pydirectinput.press('f3')
    time.sleep(3)

def screen_results(model):
    # Define current screenshot as source
    source = 'screen'
    # Run inference on the source
    results = model(source, stream=True)  # list of Results objects
    process_results(results)

def process_results(results):
    lock.acquire()
    for result in results:
        boxes = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        try:
            if pyautogui.locateOnScreen('peso.png',confidence=0.990):  # Adjust confidence as neededd
                autosell()
        except pyautogui.ImageNotFoundException:
            pass  # Image not found, continue to the next condition

        try: 
            if pyautogui.locateOnScreen('fechar.png',confidence=0.700):
                pyautogui.moveTo(x=604, y=904)
                time.sleep(0.2)
                pydirectinput.press('f4')
                time.sleep(0.2)
        except pyautogui.ImageNotFoundException:
            pass  # Image not found, continue to the next condition 

        if len(boxes) >= 2:
            min_distance = float('inf')
            closest_box = None
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                distance = ((screen_center_x - box_center_x) ** 2 + (screen_center_y - box_center_y) ** 2) ** 0.5

                # Adicionando a nova condição para verificar se a caixa está dentro de uma área definida
                if distance < min_distance and distance <= area_radius:
                    min_distance = distance
                    closest_box = box   
            
            if closest_box is not None:
                x1, y1, x2, y2 = closest_box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                pydirectinput.moveTo(center_x, center_y)
                pydirectinput.press('f5',presses=3,interval=0.02)
                pydirectinput.press('f6',presses=3,interval=0.02)
            else:
                pydirectinput.press('f4')

        elif len(boxes) <= 1:
            pydirectinput.press('f4')
    lock.release()


def main():
    model = YOLO('jup.pt')

    screen_results_thread = threading.Thread(target=screen_results, args=(model,))
    screen_results_thread.start()

    # No need to create a separate thread for process_results anymore

if __name__ == "__main__":  
    main()