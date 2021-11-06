import cv2

# our image
# img_file = "C:\\Users\\asus\\Pictures\\photos\\car2.jpg"
# video = cv2.VideoCapture("D:\\minor project\\myproj\\cars.mp4")
video = cv2.VideoCapture("D:\minor project\myproj\pedestrians.mp4")

# our pre-trained car classifier and pedestrian classifier
car_tracker_file = "D:\\minor project\\myproj\\car_model.xml"
pedestrian_tracker_file = "D:\\minor project\\myproj\\pedestrian_model.xml"


# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# run forever until car stops or something or crashes
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # must convert to grey scale 
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangules around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)


    # display the image with the faces spotted
    cv2.imshow('Dhy Car Detector', frame)

    # listen for a key press for 1millisec. then move on
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the video capture
video.release()
    

"""
# create an open cv image
img = cv2.imread(img_file)

# convert it to grey scale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# draw rectangules around cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)


# display the image with the face spotted
cv2.imshow('Dhy Car Detector', img)

# don't autoclose(wait here in the code and listen for a key press)
cv2.waitKey()

print ("Code completed")
"""