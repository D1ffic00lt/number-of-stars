import cv2


def write_to_file(data, file):
    f = open(file, 'a', encoding='utf8')
    f.writelines(data)
    f.close()


class Recogniser:
    def __init__(self, path, Index):
        self.image = cv2.imread(path)
        self.approx = None
        self.c_area = None
        self.peri = None
        self.hierarchy = None
        self.holes_area = None
        self.area = None
        self.contours = None
        self.closed = None
        self.kernel = None
        self.index = Index
        self.gray = None
        self.edged = None

    def findContours(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.GaussianBlur(self.gray, (7, 7), 1)
        self.edged = cv2.Canny(self.gray, 200, 400, apertureSize=5, L2gradient=True)
        cv2.imwrite("output_photos/edged{0}.jpg".format(str(self.index)), self.edged)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.closed = cv2.morphologyEx(self.edged, cv2.MORPH_CLOSE, self.kernel)

        return cv2.findContours(self.closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    def count(self, data):
        print(Recogniser.count_by_area(self, self.image, data))

    def count_by_area(self, image, data):
        self.contours, self.hierarchy = data
        self.area = cv2.contourArea(self.contours[0])
        self.holes_area = 0

        for c in self.contours:
            self.c_area = cv2.contourArea(c)
            if self.c_area < self.area / 15:
                self.holes_area += self.c_area
            self.peri = cv2.arcLength(c, True)
            self.approx = cv2.approxPolyDP(c, 0.005 * self.peri, True)
            cv2.drawContours(image, [self.approx], -1, (0, 255, 0), 4)
        return len(self.contours)


photos = [
    "IMG_6407.PNG",
    "Ночное_небо_Таганая.jpg"
]
index = 0

for photo in photos:
    recogniser = Recogniser("input_photos/" + photo, index)
    recogniser.count(recogniser.findContours())
    index += 1
