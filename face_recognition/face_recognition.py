from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import zipfile
from retinaface import RetinaFace

img1 ='022_9b0e7dc8.jpg'

faces=RetinaFace.detect_faces(img1)
img_array = mpimg.imread(img1) # Read the image once outside the loop

cropped_face_path='cropped_face.jpg'

for face_data in faces.values():
    # Extract bounding box coordinates
    x1, y1, x2, y2 = face_data['facial_area'] 
    
    # Crop the face from the original image
    face_img = img_array[y1:y2, x1:x2] 
    plt.imsave(cropped_face_path,face_img)
    plt.imshow(face_img) # Display the cropped face
    plt.show()


image1=cropped_face_path

archive_path = "archive.zip"
db_extracted_path = r"C:/Users/tragu/Downloads/archive_extracted"
model_name="VGG-Face"
def unzip_archive(archive_path, extract_to):
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def recognition(image, db_path):
    if not os.path.exists(db_path):
      unzip_archive(archive_path, db_extracted_path)

    dfs = DeepFace.find(img_path=image, db_path=db_extracted_path, model_name="ArcFace")
    
    # Display the input image
    img = mpimg.imread(image)
    plt.imshow(img)
    filename=os.path.basename(image)
    plt.title(f"Input Image:{filename}")
    plt.show()

    # Display the recognized image
    if  len(dfs) > 0:#check if any face were recognized
        recognized_image_path = dfs[0]['identity'][0] #extract the string file path
        recognized_img = mpimg.imread(recognized_image_path)
        filename=os.path.basename(recognized_image_path)
        plt.imshow(recognized_img)
        plt.title(f"Recognized Image:{filename}")
        plt.show()
    
    return dfs

# Call the recognition function
recognition(image1, db_extracted_path)

