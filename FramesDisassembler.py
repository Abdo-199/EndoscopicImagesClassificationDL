import cv2
import os
import openpyxl
import argparse

def extract_frames(video_path, output_folder, excel_path, device, quality, organ,frame_rate:int):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames_data = []
    frame_number = 0
    written_frame_number = 0 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if not frame_number % frame_rate == 0 and written_frame_number != 0:
            continue

        video_name = os.path.basename(video_path).split(".",1)[0]
        output_path = os.path.join(output_folder, f"{video_name}_{written_frame_number:04d}.png")
        cv2.imwrite(output_path, frame)
        frames_data.append({"Frame": output_path, "Device": device, "Quality": quality, "Organ": organ})
        written_frame_number += 1

    cap.release()

    workbook = openpyxl.load_workbook(excel_path)
    worksheet = workbook.active
    for row_data in frames_data:
        row = [row_data["Frame"], row_data["Device"], row_data["Quality"], row_data["Organ"]]
        worksheet.append(row)
    workbook.save(excel_path)
    os.rename(video_path,output_folder + "\\" +os.path.basename(video_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video and save them as images.')
    parser.add_argument('video_path', type=str, help='path to the input video file')
    parser.add_argument('output_folder', type=str, help='path to the output folder')
    parser.add_argument('excel_path', type=str, help='path to the output Excel file')
    parser.add_argument('device', type=str, help='The device name')
    parser.add_argument('quality', type=str, help='the quality of the frame')
    parser.add_argument('organ', type=str, help='Organ')
    parser.add_argument('frame_rate', type=str, help='Just the frames which are divisable by the int you inter will be saved')
    args = parser.parse_args()

    video_path = args.video_path
    output_folder = args.output_folder
    excel_path = args.excel_path
    device = args.device
    quality = args.quality
    organ = args.organ
    frame_rate = int(args.frame_rate)

    extract_frames(video_path, output_folder, excel_path, device, quality, organ, frame_rate)
