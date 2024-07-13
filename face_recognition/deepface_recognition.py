from deepface import DeepFace
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import zipfile
from retinaface import RetinaFace


image1 = 'flask/url building using flask/022_9b0e7dc8.jpg'

faces=RetinaFace.detect_images(imagepath='img',align=True)

plt.imshow(faces)
plt.show()

archive_path = r"C:\Users\tragu\Downloads\flask\url building using flask\archive.zip"
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
