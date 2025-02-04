from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

path_folder = '/mnt/disks/location/input_train'
path_csv = '/mnt/disks/location/Y_train.csv'

# Dict defining rotation and cropping depending on component
rot_crop_data = {
        "Die01":[55,(340, 120, 500, 680)], # (left, upper, right, lower)
        "Die02":[-44, (480, 210, 640, 930)],
        "Die03":[134, (460, 200, 620, 920)],
        "Die04":[35, (310, 130, 470, 690)]
}

def rotate_and_crop_image(path, angle, crop_box, plot=False):
    """
    Applique une rotation d'un angle donné et un crop sur une image, puis enregistre le résultat en .png.
    Optionnellement, affiche l'image si plot=True.
    
    Arguments :
    - input_image_path : Chemin de l'image d'entrée.
    - output_image_path : Chemin pour enregistrer l'image modifiée.
    - angle : L'angle de rotation en degrés (positif dans le sens antihoraire).
    - crop_box : Un tuple (left, upper, right, lower) définissant les points du crop en pixels.
    - plot : Booléen, si True affiche l'image modifiée.
    """
    # Ouvrir l'image
    image = Image.open(path)
    
    # Appliquer la rotation (expand=True permet d'agrandir l'image pour qu'elle s'ajuste au cadre après rotation)
    rotated_image = image.rotate(angle, expand=True)
    
    # Appliquer le crop avec les points fournis
    cropped_image = rotated_image.crop(crop_box)
    
    # Enregistrer l'image au format PNG
    cropped_image.save(path, format='PNG')
    #print(f"Image enregistrée avec succès sous {path}")
    
    # Si plot=True, afficher l'image modifiée
    if plot:
        plt.imshow(cropped_image)
        plt.axis('off')  # Masquer les axes
        plt.title(f"Image après rotation de {angle}° et crop")
        plt.show()


test_df = pd.read_csv(path_csv)

for index, row in test_df.iterrows():
   path = os.path.join(path_folder, row['filename'])
   rotate_and_crop_image(path, rot_crop_data[row['lib']][0], rot_crop_data[row['lib']][1], plot=False)

print(f"successfully rotated {path_folder} ")