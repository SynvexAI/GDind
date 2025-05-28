import time
import cv2
import numpy as np
import mss

# Настрой сюда вручную координаты своей игры (без рамок)
FALLBACK_GAME_REGION = {"top": 40, "left": 0, "width": 800, "height": 600}

def main():
    sct = mss.mss()
    region = FALLBACK_GAME_REGION
    print(f"[INFO] Capture region: {region}")

    last_time = time.time()
    frame_count = 0

    try:
        while True:
            start = time.time()
            sct_img = sct.grab(region)
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            
            cv2.imshow("Capture Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Простой подсчёт FPS
            frame_count += 1
            if time.time() - last_time >= 1.0:
                print(f"[FPS] {frame_count}")
                frame_count = 0
                last_time = time.time()

            # Защита от слишком быстрых итераций
            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
