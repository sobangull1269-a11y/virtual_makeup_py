import cv2
from utils import apply_makeup, FOUNDATION_SHADES

# Video Input from Webcam
video_capture = cv2.VideoCapture(0)

print("Foundation Shades Available:")
for shade_id, shade_info in FOUNDATION_SHADES.items():
    print(f"Shade {shade_id}: {shade_info['name']} ({shade_info['undertone']})")

print("\nControls:")
print("Press 3-8: Select foundation shade (03, 04, 05, 06, 07, 08)")
print("Press 'f': Foundation mode")
print("Press 'l': Lips mode") 
print("Press 'b': Blush mode")
print("Press 'n': No makeup")
print("Press ESC: Exit")

current_feature = 'foundation'
current_shade = '03'

while True:
    ret_val, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    if ret_val:
        cv2.imshow("Original", frame)
        
        # Apply makeup with current shade as parameter
        feat_applied = apply_makeup(frame, True, current_feature, False, current_shade) 
        cv2.imshow("Virtual Makeup", feat_applied)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif ord('3') <= key <= ord('8'):
            shade_key = f"0{key - ord('0')}"  # Convert to '03', '04', etc.
            if shade_key in FOUNDATION_SHADES:
                current_shade = shade_key
                current_feature = 'foundation'
                print(f"Switched to shade: {FOUNDATION_SHADES[shade_key]['name']}")
        elif key == ord('f'):
            current_feature = 'foundation'
            print("Foundation mode")
        elif key == ord('l'):
            current_feature = 'lips'
            print("Lips mode")
        elif key == ord('b'):
            current_feature = 'blush'
            print("Blush mode")
        elif key == ord('n'):
            current_feature = 'none'
            print("No makeup mode")

video_capture.release()
cv2.destroyAllWindows()