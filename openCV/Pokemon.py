import cv2

class Pokemon:
    def __init__(self, name, image_path, gx1, gy1):
        self.name = name
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.image = cv2.resize(self.image, (150, 150))
        self.h, self.w = self.image.shape[:2]

        # Spawn position (on top of grass)
        self.x1 = gx1 + 75
        self.y1 = gy1 - self.h
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

        # Start invisible
        self.image[:, :, 3] = 0