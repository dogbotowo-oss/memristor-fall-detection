import os

import cv2
import numpy as np


source_path = r"F:\12.18-2\zenodo_falldb_video_split\train\Fall\person1a_11.mp4"
output_dir = r"F:\12.18-2\论文插图\person1a_11_16frames"

os.makedirs(output_dir, exist_ok=True)
capture = cv2.VideoCapture(source_path)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
start_frame = max(0, frame_count // 2 - 8)
capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames = []
for _ in range(16):
    success, frame = capture.read()
    if not success:
        break
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA))
capture.release()

for index, frame in enumerate(frames, 1):
    cv2.imwrite(os.path.join(output_dir, f"frame_{index:02d}.png"), frame)

contact_sheet = np.full((1200, 1200, 3), 255, dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
for index, frame in enumerate(frames, 1):
    row, column = divmod(index - 1, 4)
    image = cv2.cvtColor(cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    top = row * 300 + 30
    left = column * 300 + 30
    contact_sheet[top:top + 256, left:left + 256] = image
    cv2.rectangle(contact_sheet, (left, top), (left + 256, top + 256), (50, 50, 50), 2)
    cv2.putText(contact_sheet, f"t{index}", (left, top + 282), font, 0.85, (30, 30, 30), 2, cv2.LINE_AA)

cv2.imwrite(os.path.join(output_dir, "contact_sheet_4x4.png"), contact_sheet)
with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as handle:
    handle.write(f"Source: {source_path}\n")
    handle.write(f"Frame range: {start_frame}-{start_frame + len(frames) - 1}\n")
    handle.write(f"FPS: {fps}\n")
    handle.write("Input size: 16 x 1 x 64 x 64\n")
