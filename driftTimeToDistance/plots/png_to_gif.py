import imageio
import os

def main():
    image_folder = 'driftTimeToDistance\plots\\gif_raw'
    output_file = 'driftTimeToDistance\plots\gif\\fit_val.gif'

    # Get a list of all the PNG files in the folder
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

    # Sort the images in ascending order
    images.sort()

    # Read the images and append them to a list
    gif_images = []
    for image in images:
        gif_images.append(imageio.imread(image))

    # Save the list of images as a GIF
    imageio.mimsave(output_file, gif_images, duration=0.2)
if __name__ == "__main__":
    main()