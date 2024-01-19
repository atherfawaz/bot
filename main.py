import os

import requests


def download_file_from_url(file_url, destination_path, file_name):
    # Create the destination directory if it doesn't exist
    destination_directory = os.path.dirname(destination_path)
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    response = requests.get(file_url)

    if response.status_code == 200:
        with open(f"{destination_path}/{file_name}", "wb") as file:
            file.write(response.content)
        print(f"Downloaded file to: {destination_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


file_url = "https://drive.google.com/file/d/1cnVsGUDSmRbhUR_8WLYwo4C2y8g2ctqF/view?usp=drive_link"
file_name = "index.faiss"
destination_path = "./faiss/"
download_file_from_url(file_url, destination_path, file_name)


file_url = "https://drive.google.com/file/d/12UqrIrfAgEyY-wMwV2VC46b3Gx-L09_w/view?usp=drive_link"
file_name = "index.pkl"
destination_path = "./faiss/"
download_file_from_url(file_url, destination_path, file_name)
