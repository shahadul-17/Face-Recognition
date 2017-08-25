import cv2
import os
import numpy
import time

class FaceRecognitionEngine:

    dodgerBlueColor = None
    faceCascadeClassifier = None
    webcam = None
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.isRunning = False
        self.webcamImage = None
        global dodgerBlueColor, faceCascadeClassifier, webcam

        dodgerBlueColor = (255, 144, 30)
        faceCascadeClassifier = cv2.CascadeClassifier("data\\face-cascade-classifier.xml")
        webcam = cv2.VideoCapture(0)
    
    def loadImage(self, fileName):
        return cv2.imread(fileName)

    def convertToGray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detectFaces(self, image):
        global faceCascadeClassifier

        return faceCascadeClassifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def countFaces(self, faces):
        return len(faces)
    
    def markFaces(self, faces, image):
        global dodgerBlueColor

        for (x, y, width, height) in faces:
            cv2.rectangle(image, (x, y), (x + width, y + height), dodgerBlueColor, 2)
    
        return image

    def cropFaces(self, faces, image):
        croppedImages = []      # an empty list of cropped images...

        for (x, y, width, height) in faces:
            croppedImages.insert(len(croppedImages), image[y:(y + height), x:(x + width)])
        
        return croppedImages

    def showImage(self, image):
        cv2.imshow("", image)
        
        if cv2.waitKey(1) == 27:
            self.stopWebcam()
    
    def createFaceRecognizer(self):
        return cv2.face.createLBPHFaceRecognizer()

    def capture(self):
        image = self.webcamImage
        grayImage = self.convertToGray(image)
        faces = self.detectFaces(grayImage)
        facesDetected = self.countFaces(faces)
        
        if facesDetected > 0:
            image = self.cropFaces(faces, image)[0]
            cv2.imwrite("1.jpg", image)

    def resizeData(self, imagePath):
        image = self.loadImage(imagePath)

        cv2.imshow("", image)
        cv2.waitKey(0)

        grayImage = self.convertToGray(image)
        faces = self.detectFaces(grayImage)
        facesDetected = self.countFaces(faces)

        print facesDetected

        if facesDetected > 0:
            images = self.cropFaces(faces, image)
            cv2.resize(images[0], (200, 200))
            cv2.imwrite(imagePath, images[0])
            cv2.imshow("", images[0])
            cv2.waitKey(0)

    def buildDatabase(self):
        print "initializing..."

        ID = 0
        facesDetected = 0
        subdirectoryNames = os.listdir("data\\faces")
        listIDs = []
        listFaces = []

        faceRecognizer = self.createFaceRecognizer()

        print "organizing database..."

        for subdirectoryName in subdirectoryNames:
            files = os.listdir("data\\faces\\" + subdirectoryName)

            for _file in files:
                image = self.loadImage("data\\faces\\" + subdirectoryName + "\\" + _file)
                grayImage = self.convertToGray(image)
                faces = self.detectFaces(grayImage)
                facesDetected = self.countFaces(faces)

                print "faces detected = ", facesDetected

                if facesDetected == 1:      # the image must contain only one face...
                    print ID,"data\\faces\\" + subdirectoryName + "\\" + _file
                    listIDs.append(ID)
                    croppedImage = self.cropFaces(faces, grayImage)[0]
                    listFaces.append(numpy.array(croppedImage, 'uint8'))
                
                facesDetected = 0
            # loop ends here...
            
            ID += 1
        # loop ends here...

        faceRecognizer.train(listFaces, numpy.array(listIDs))
        faceRecognizer.save("data//face-recognition-database.xml")

        print "operation completed successfully..."

        '''for _file in files:
            image = self.loadImage(directoryName + "\\" + _file)
            grayImage = self.convertToGray(image)
            faces = self.detectFaces(grayImage)
            facesDetected = self.countFaces(faces)

            if facesDetected == 1:
                croppedImages = self.cropFaces(faces, grayImage)

                for croppedImage in croppedImages:
                    # cv2.imwrite("data\\" + personName + "\\" + str(i) + ".jpg", croppedImage)
                    self.showImage(croppedImage)
            
                i += 1'''

    '''def learnFace(self, directory, personName):
        IDs = []
        faces = []
        directories = [os.path.join(directory, item) for item in os.listdir(directory)]
        faceRecognizer = self.createFaceRecognizer()

        for imagePath in directories:
            IDs.append(int(imagePath[imagePath.rfind('\\') + 1:].split('.')[0]))
            faces.append(numpy.array(self.convertToGray(self.loadImage(imagePath)), 'uint8'))
        
        faceRecognizer.train(faces, numpy.array(IDs))
        faceRecognizer.save("data//trainingdata.xml")
    '''
    # def resizeImage(self, fileName):
        # return cv2.resize(self.loadImage(fileName), (200, 200))

    def recognizeFace(self, image):
        names = os.listdir("data\\faces")
        faceRecognizer = self.createFaceRecognizer()
        faceRecognizer.load("data\\face-recognition-database.xml")

        # image = self.loadImage(imagePath)
        grayImage = self.convertToGray(image)
        faces = self.detectFaces(grayImage)
        facesDetected = self.countFaces(faces)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image,'Shahrukh Khan', ((x + width) / 2, 130), font, 1, dodgerBlueColor)
        # self.showImage(image)

        if facesDetected > 0:
            for (x, y, width, height) in faces:
                self.x = x
                self.y = y
                self.width = width
                self.height = height

                ID, confidence = faceRecognizer.predict(grayImage[y:(y + height), x:(x + width)])

                if confidence < 50.0:
                    print "The person is '" + names[ID] + "' with ID = " + str(ID) + " with confidence = " + str(confidence)
                else:
                    print "The person is Unknown..."
        
        return image

    def startWebcam(self):
        self.isRunning = True

        while self.isRunning:
            returnValue, self.webcamImage = webcam.read(0)
            cv2.rectangle(self.webcamImage, (self.x, self.y), (self.x + self.width, self.y + self.height), dodgerBlueColor, 2)   # marks faces...
            self.showImage(self.webcamImage)
    
    def stopWebcam(self):
        self.isRunning = False
        cv2.destroyAllWindows()