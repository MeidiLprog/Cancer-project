# Cancer-project

Ce projet a pour objectif d’entraîner un réseau de neurones convolutionnel (CNN) pour détecter la présence d’un cancer du sein à partir d’images médicales.  
Il a été réalisé avec Python et TensorFlow/Keras dans un but d’apprentissage.

---

## Données

Les données proviennent du dépôt public :

https://github.com/MachineLearnia/breast_cancer_public_data

Structure utilisée :

```
breast_cancer_public_data/
└── data_2/
    ├── Cancer/
    └── Negative/
```

Chaque image est chargée puis redimensionnée à :

224 × 224 × 3

Les classes sont :

Negative = 0, Cancer = 1

---

## Préparation des données

- Chargement des images
- Redimensionnement
- Association image/label
- Découpage en train/test (20% test)

```python
classes = ["Negative","Cancer"]
dataset = []
for class_label in classes:
    class_path = os.path.join(folder_path, class_label)
    label_index = classes.index(class_label)
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224))
        dataset.append([img,label_index])
```

---

## Architecture du modèle

```python
model = Sequential()
model.add(Conv2D(100,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(100,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

---

## Entraînement

```python
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)
```

---

## Prérequis

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

---

## Avertissement

Ce projet est uniquement destiné à des fins pédagogiques.  
Il ne constitue pas un outil médical et ne doit pas être utilisé comme méthode de diagnostic.
