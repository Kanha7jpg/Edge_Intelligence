import cv2
import os

class StimulusLoader:
    def __init__(self, source):
        """
        Initializes the stimulus loader.
        :param source: Can be a path to a directory of images or a video file.
        """
        self.source = source
        self.is_video = False
        self.cap = None
        self.image_files = []
        self.current_idx = 0
        
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Live Webcam Feed
            self.is_video = True
            self.cap = cv2.VideoCapture(int(source))
        elif os.path.isdir(source):
            self.image_files = sorted([os.path.join(source, f) for f in os.listdir(source) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        elif os.path.isfile(source):
            self.is_video = True
            self.cap = cv2.VideoCapture(source)
        else:
            raise ValueError("Invalid source path. Must be a directory, video file, or camera index (e.g., 0).")

    def get_next_frame(self):
        """
        Yields the next frame in the stimulus sequence.
        Returns None when finished.
        """
        if self.is_video:
            if not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                return None
        else:
            if self.current_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.current_idx])
                self.current_idx += 1
                return frame
            else:
                return None
                
    def reset(self):
        self.current_idx = 0
        if self.is_video and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
