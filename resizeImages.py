import os
from PIL import Image


def resize_and_rename_images(input_folder, output_folder, target_size=(100, 100)):
    # Preveri, če mapa za izhod obstaja, če ne, jo ustvari
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Pridobi vse datoteke v mapi
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    # Iteriraj skozi vse slike in jih predelaj
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)

        # Odpri sliko
        with Image.open(image_path) as img:
            # Spremeni velikost slike, brez obrezovanja
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

            # Shranjevanje nove slike z novim imenom
            output_path = os.path.join(output_folder, f"{idx}.jpg")
            img_resized.save(output_path, "JPEG")

    print(f"Uspešno pretvorjenih {len(image_files)} slik v mapo: {output_folder}")


# Primer uporabe
if __name__ == "__main__":
    input_folder = "data"  # Zamenjaj z dejansko potjo do mape z vhodnimi slikami
    output_folder = "dataset"  # Zamenjaj z želeno potjo za shranjevanje obdelanih slik
    resize_and_rename_images(input_folder, output_folder)