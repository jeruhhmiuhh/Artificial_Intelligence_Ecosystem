from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import matplotlib.pyplot as plt
import os

def apply_ethereal_glow(image_path, output_path="ethereal_output.png"):
    try:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((512, 512))

        # boost saturation for a vibrant base
        saturated = ImageEnhance.Color(img_resized).enhance(1.8)

        # create a strong blur layer for the glow
        glow_layer = saturated.filter(ImageFilter.GaussianBlur(radius=12))

        # blend the glow back onto the saturated image using screen blending
        blended = ImageChops.screen(saturated, glow_layer)

        # second softer glow pass for extra dreaminess
        soft_glow = blended.filter(ImageFilter.GaussianBlur(radius=4))
        final = ImageChops.screen(blended, soft_glow)

        # boost brightness slightly
        final = ImageEnhance.Brightness(final).enhance(1.1)

        # save and show side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(final)
        axes[1].set_title("Ethereal Glow")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        print(f"Ethereal glow saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Ethereal Glow Filter (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_ethereal.png"
        apply_ethereal_glow(image_path, output_file)