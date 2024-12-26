import cv2
import os
import numpy as np

dataPath = r"C:\Users\deivi\OneDrive\Escritorio\semana12\data"
peopleList = os.listdir(dataPath)
print("Lista de personas: ", peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + "/" + nameDir
    print("Analizando imágenes")
    for fileName in os.listdir(personPath):
        print("Caras: ", nameDir + "/" + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + "/" + fileName, 0))
        image = cv2.imread(personPath + "/" + fileName, 0)
       # cv2.imshow("image", image)
       # cv2.waitKey(10)

label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Entrenando")

face_recognizer.train(facesData, np.array(labels))


# Almacenando el modelo obtenido
face_recognizer.write("modeloEigenFace.xml")
print("Modelo almacenado")
