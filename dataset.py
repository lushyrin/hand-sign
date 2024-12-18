import cv2
import os

# Setup
capture = cv2.VideoCapture(0)  # Use 0 for the default camera
sign_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
current_sign = 0
save_dir = 'asl_dataset'
os.makedirs(save_dir, exist_ok=True)

print("Press 'n' to change the sign, 's' to save an image, and 'q' to quit.")

# Main loop
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Capture ASL Signs', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the frame
        sign_dir = os.path.join(save_dir, sign_names[current_sign])
        os.makedirs(sign_dir, exist_ok=True)
        img_name = os.path.join(sign_dir, f'{sign_names[current_sign]}_{len(os.listdir(sign_dir))}.png')
        cv2.imwrite(img_name, frame)
        print(f'Saved {img_name}')
    elif key == ord('n'):
        # Move to the next sign
        current_sign = (current_sign + 1) % len(sign_names)
        print(f'Current sign: {sign_names[current_sign]}')
    elif key == ord('q'):
        # Quit the loop
        break

# Clean up
capture.release()
cv2.destroyAllWindows()
