import datetime
import cv2


def movie_info(path):
    vid_obj = cv2.VideoCapture(path)
    if vid_obj.isOpened():
        frame_number = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Total number of frames: {:.0f}'.format(frame_number))

        fps = int(vid_obj.get(cv2.CAP_PROP_FPS))
        print('Frames per Second: {:.0f}'.format(fps))

        height = int(vid_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(f'Width X Height: {width :.0f} x {height :.0f}')
        print('Width X Height: {:.0f} x {:.0f}'.format(width, height))

        seconds = frame_number / fps
        video_time = str(datetime.timedelta(seconds=seconds))
        print('This film is {} seconds long - {}'.format(seconds, video_time))

        return frame_number, fps, width, height, seconds
