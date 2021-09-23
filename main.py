import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
# this library is for video file
# from imutils.video import FileVideoStream
from imutils.video import VideoStream
from playsound import playsound
import imutils
import datetime
import time
import dlib
# import libraries for accesing the data base to update data
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

DEVICE_ID = 'deviceID8'

cred = credentials.Certificate("serviceAccountKeyDrivital.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


def find_owner_key():
    persons = db.collection('persons').get()
    for person in persons:
        person_key = person.id
        print(person_key)
        devices = db.collection('persons').document(person_key).collection('devices').get()
        for device in devices:
            device_key = device.id
            if (device_key == DEVICE_ID):
                return person_key


owner_id = find_owner_key()
print(owner_id)


def add_event(event_type, event_date: datetime.datetime):
    device = db.collection('persons').document(owner_id).collection('devices').document(DEVICE_ID).get()
    event_list = device.to_dict()['event_list']
    event_list.append({'date': event_date, 'type': event_type})
    db.collection('persons').document(owner_id).collection('devices').document(DEVICE_ID).set(
        {'event_list': event_list}, merge=True)


MAX_ALARM_FRAMES_DURATION = 120  # maximum alarm length to wait between alarms
YAWN_THRESH = 20
BLINK_TIME_RATIO = 0.5
current_blink_time_ratio = 0  # the ratio between the no of blinks and the time that have passed
alarm_on = False  # boolean shows if there is an alarm on or not
count_frames = False  # boolean to know if to count the frames or not
frames_nr = 0  # total number of frames since the last alarm was given
no_face_frames = 0  # consecutive number of frames in which no face was found
total_number_of_frames = 0  # total number of frames that have passed while the faced was recognized
real_number_of_frames = 0  # total number of frames that have passsed form the moment the device was turned on
time_blinked_to_often = 0  # the last time in seconds that the person blinked too often
last_yawn = - 10  # the last time in seconds that the person yawned
last_absence = 0  # the last time in seconds that the person was absent
last_closed_eyes = 0  # the last time in seconds that the person had their eyes closed


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    # return the eye aspect ratio`
    return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 33
EYE_AR_CONSEC_FRAMES_2 = 1

# initialize the frame counters and the total number of blinks
counter = 0
counter2 = 0
total2 = 0


def sound_alarm(path):
    # play an alarm sound
    playsound(path)


def lip_distance(shape):
    # returns the distance between the lips
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
# vs = VideoStream(usePiCamera= True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    total_number_of_frames += 1
    real_number_of_frames += 1
    real_seconds_number = real_number_of_frames // 11
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    if count_frames == True:
        frames_nr += 1

    # NO FACE DETECTION
    if len(rects) == 0:
        total_number_of_frames -= 1
        cv2.putText(frame, 'No face detected', (400, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 0), 1)
        no_face_frames += 1
        if no_face_frames >= 60:
            if real_seconds_number - last_absence >= 30:
                # add absence event in the data base
                add_event('absence', datetime.datetime.now())

                # update last_absence
                last_absence = real_seconds_number
            if alarm_on == False:
                alarm_on = True
                count_frames = True
                frames_nr = 0

                # check to see if an alarm file was supplied,
                # and if so, start a thread to have the alarm
                # sound played in the background
                t = Thread(target=sound_alarm,
                           args=('alarmsAbsence/alarm_face_not_detected.mp3',))
                t.deamon = True
                t.start()

    # otherwise a face is detected and we don't play the alarm
    # so reset the alarm to false
    else:
        no_face_frames = 0
        if frames_nr >= MAX_ALARM_FRAMES_DURATION:
            alarm_on = False

    # get the number of seconds
    seconds = total_number_of_frames // 11

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # YAWNING
        # distance between lips
        lip_dist = lip_distance(shape)
        cv2.putText(frame, str(lip_dist), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if lip_dist > YAWN_THRESH:
            cv2.putText(frame, 'Yawning', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # if it passed at least a minute since the last yawn update the data base
            if seconds - last_yawn >= 10:
                # add yawn event in the data base
                add_event('yawn', datetime.datetime.now())

                # update the last_yawn variable
                last_yawn = seconds

            # if the alarm is not on, turn it on
            if alarm_on == False:
                alarm_on = True
                count_frames = True
                frames_nr = 0

                t2 = Thread(target=sound_alarm,
                            args=('alarms/alarm_drowsiness.mp3',))
                t2.deamon = True
                t2.start()

        else:
            if frames_nr >= MAX_ALARM_FRAMES_DURATION:
                alarm_on = False

        # EYES CLOSED and NUMBER OF BLINKS IN TIME INTERVAL
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            counter += 1

            # if the eyes were closed for a sufficient number of frames
            # then sound the alarm
            if counter >= EYE_AR_CONSEC_FRAMES:
                if seconds - last_closed_eyes >= 5:
                    # create closed eyes event in the data base
                    add_event('closed_eyes', datetime.datetime.now())

                    # update last_closed_eyes variable
                    last_closed_eyes = seconds
                # if the alarm is not on, turn it on
                if alarm_on == False:
                    alarm_on = True
                    count_frames = True
                    frames_nr = 0
                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background

                    t3 = Thread(target=sound_alarm,
                                args=('alarms/alarm_drowsiness.mp3',))
                    t3.deamon = True
                    t3.start()
                # draw an alarm on the frame
                cv2.putText(frame, "EYES CLOSED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            counter = 0
            if frames_nr >= MAX_ALARM_FRAMES_DURATION:
                alarm_on = False

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            counter2 += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of frames
            # then increment the total number of blinks
            if counter2 >= EYE_AR_CONSEC_FRAMES_2:
                total2 += 1

            # reset the eye frame counter
            counter2 = 0

        # CHECK IF BLINK_TIME RATIO IS ABOVE THE LIMIT
        if seconds >= 60:
            current_blink_time_ratio = total2 / seconds
            if current_blink_time_ratio > BLINK_TIME_RATIO:
                # if the alarm is not on and it passed 1 minute from the last message
                # turn on the alarm, and create often_blink event in the data base
                if alarm_on == False and seconds - time_blinked_to_often >= 60:
                    add_event('often_blink', datetime.datetime.now())
                    time_blinked_to_often = seconds
                    alarm_on = True
                    count_frames = True
                    frames_nr = 0

                    t4 = Thread(target=sound_alarm,
                                args=('alarms/alarm_drowsiness.mp3',))
                    t4.deamon = True
                    t4.start()
                # draw an alarm on the frame
                cv2.putText(frame, "BLINKED TOO MANY TIMES!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                if frames_nr >= MAX_ALARM_FRAMES_DURATION:
                    alarm_on = False

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(total2), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw the total number of seconds that have passed
        # since the project was started
        cv2.putText(frame, "Seconds: {}".format(seconds), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
