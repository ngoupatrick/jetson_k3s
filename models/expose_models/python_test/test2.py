# python3 /home/patrick/Bureau/vpn/jetson_k3s/models/expose_models/python_test/test2.py http://picomaster:31080/predict /home/patrick/Bureau/test/water-citron.jpg
import requests
import sys
import os

def main():
    # Vérification des arguments
    if len(sys.argv) < 3:
        print("Usage: python3 predict_client.py <URL> <IMAGE_PATH>")
        print("Exemple: python3 predict_client.py http://@IPM:30080/predict test.jpg")
        sys.exit(1)

    url = sys.argv[1]
    image_path = sys.argv[2]

    # Vérification de l'existence du fichier
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier '{image_path}' est introuvable.")
        sys.exit(1)

    try:
        # Envoi de la requête POST
        with open(image_path, "rb") as img_file:
            resp = requests.post(url, files={"image": img_file})
            
        # Affichage du résultat JSON
        if resp.status_code == 200:
            print(resp.json())
        else:
            print(f"Erreur {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'appel : {e}")

if __name__ == "__main__":
    main()
