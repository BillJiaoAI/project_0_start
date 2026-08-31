import ctypes
import random
import time

user32 = ctypes.windll.user32
VK_CODE = {'3': 0x33, '5': 0x35}

def press_key(key):
    user32.keybd_event(VK_CODE[key], 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(VK_CODE[key], 0, 2, 0)

def is_key_pressed(key):
    return user32.GetAsyncKeyState(VK_CODE[key]) & 0x8000 != 0

def main():
    print("自动按3脚本启动...")
    print("每0.5s-1s随机按下3键")
    print("按下5键停止脚本")
    
    while True:
        if is_key_pressed('5'):
            print("检测到5键，停止脚本")
            break
        
        press_key('3')
        delay = random.uniform(0.5, 1.0)
        print(f"已按下3键，下次按键在 {delay:.2f}s 后")
        
        start_time = time.time()
        while time.time() - start_time < delay:
            if is_key_pressed('5'):
                print("检测到5键，停止脚本")
                return
            time.sleep(0.01)

if __name__ == "__main__":
    main()