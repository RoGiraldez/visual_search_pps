#ESTE NUNCA FUE UTILIZADO

from tensorflow.keras.preprocessing import image
from tensorflow.keras import Input
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from SpatialPyramidPooling import SpatialPyramidPooling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import autokeras as ak


base_model = MobileNet(weights='imagenet', include_top=False, input_shape=[None, None, 3])
x = SpatialPyramidPooling([1, 2, 4])(base_model.output)

# create generator
datagen = ImageDataGenerator()
# prepare an iterators for each dataset
train_it = datagen.flow_from_directory(r'C:\Users\Rocío\Documents\TESIS\2da BD\WHU-RS19\train', class_mode='binary')
val_it = datagen.flow_from_directory(r'C:\Users\Rocío\Documents\TESIS\2da BD\WHU-RS19\validation', class_mode='binary')
#test_it = datagen.flow_from_directory('data/test/', class_mode='binary')

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

"""model = Model(inputs=base_model.input, outputs=x)
model.summary()"""
data_dir = r'C:\Users\Rocío\Documents\TESIS\2da BD\WHU-RS19 junto\train'
base_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
base_model.fit_generator(
    train_it,
    steps_per_epoch=20,
    epochs=50,
    validation_data=val_it,
   )

"""
print(data_dir)
batch_size = 1
img_height = 600
img_width = 600

train_data = ak.image_dataset_from_directory(
    data_dir,
    # Use 20% data as testing data.
    validation_split=0.2,
    subset="training",
    # Set seed to ensure the same split when loading testing data.
    seed=123,
    batch_size=batch_size,
)

test_data = ak.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=batch_size,
)
"""
"""clf = ak.ImageClassifier(overwrite=True, max_trials=1)
clf.fit(train_data, epochs=1)
print(clf.evaluate(test_data))"""

"""print(type(train_data))
print(len(train_data))
print(list(train_data.as_numpy_iterator()))


base_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
base_model.fit(x=train_data,
            steps_per_epoch=len(train_data),
            validation_data= test_data,
            validation_steps=len(test_data),
            epochs=30,
            verbose=2
)"""
"""def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in np.os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name  # extract the image array and class name


img_data, class_name = create_dataset(r'CV\Intel_Images\seg_train\seg_train')"""