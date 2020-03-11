from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import time
import cv2

# load the known faces and embeddings
print("[INFO] loading encodings...")
# data = pickle.loads(open(args["encodings"], "rb").read())
data = pickle.loads(open("encodings.pickle","rb").read())
names = []
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None

detectionMethod = "hog"

def detectFaces(minutesToDetect=0.25):
	t_end = time.time() + 60 * minutesToDetect
	# loop over frames from the video file stream
	while time.time() < t_end:
		# grab the frame from the threaded video stream
		frame = vs.read()

		# convert the input frame from BGR to RGB then resize it to have
		# a width of 750px (to speedup processing)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		rgb = imutils.resize(frame, width=750)
		r = frame.shape[1] / float(rgb.shape[1])

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input frame, then compute
		# the facial embeddings for each face
		boxes = face_recognition.face_locations(rgb,
			model=detectionMethod)
		encodings = face_recognition.face_encodings(rgb, boxes)
		
		# loop over the facial embeddings
		for encoding in encodings:
			# attempt to match each face in the input image to our known
			# encodings
			matches = face_recognition.compare_faces(data["encodings"],
				encoding)
			name = "Unknown"

			# check to see if we have found a match
			if True in matches:
				# find the indexes of all matched faces then initialize a
				# dictionary to count the total number of times each face
				# was matched
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				# loop over the matched indexes and maintain a count for
				# each recognized face face
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1
				# determine the recognized face with the largest number
				# of votes (note: in the event of an unlikely tie Python
				# will select first entry in the dictionary)
				name = max(counts, key=counts.get)
			global names
			# update the list of names
			names.append(name)
			names = list(set(names))

		# print(names)
		# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
			# rescale the face coordinates
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)

			# draw the predicted face name on the image
			cv2.rectangle(frame, (left, top), (right, bottom),
				(0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 2)

		# if the writer is not None, write the frame with recognized
		# faces todisk
		if writer is not None:
			writer.write(frame)	
		#comment the below lines until aboce return statement to use as module 
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)
		key = cv2.waitKey(1) & 0xFF
		# 	# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	return names
cv2.destroyAllWindows()	
vs.stop()
# detectFaces(time to capture video in minutes)
# print(detectFaces(0.15))
# do a bit of cleanup
