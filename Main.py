import os

from FaceRecognitionEngine import FaceRecognitionEngine

if __name__ == '__main__':
    facesDetected = 0

    faceRecognitionEngine = FaceRecognitionEngine()
    image = faceRecognitionEngine.loadImage("input-7.jpg")
    grayImage = faceRecognitionEngine.convertToGray(image)
    faces = faceRecognitionEngine.detectFaces(image)

    facesDetected = faceRecognitionEngine.countFaces(faces)

    print facesDetected, "face(s) detected..."
    
    croppedImages = faceRecognitionEngine.cropFaces(faces, image)

    if facesDetected > 0:
        for croppedImage in croppedImages:
            faceRecognitionEngine.showImage(croppedImage)
        
        image = faceRecognitionEngine.markFaces(faces, image)
        faceRecognitionEngine.showImage(image)

    # faceRecognitionEngine.learnFace("data\\shahrukh-khan", "srk")
    # faceRecognitionEngine.recognizeFace("3.jpg")
    # faceRecognitionEngine.showImage(faceRecognitionEngine.resizeImage("input-1.jpg"))
    '''imagePaths = os.listdir("inputs")
    
    for imagePath in imagePaths:
        print imagePath
        faceRecognitionEngine.recognizeFace("inputs\\" + imagePath)'''
    
    # faceRecognitionEngine.buildDatabase()