import cv2

sensor_locations_file = "graph_sensor_locations.csv"
sizes = [1, 0.75, 0.5, 1 / 3, 0.25]
suffixes = ["100", "075", "050", "030", "025"]

def concat_images(option):
    large_images = []
    small_images = []
    for size, suffix in zip(sizes, suffixes):
        large_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-large-{suffix}.png"))
        small_images.append(cv2.imread(f"images/{option[4]}/{option[2]}-small-{suffix}.png"))
    large_image = cv2.vconcat(large_images)
    small_image = cv2.vconcat(small_images)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-large.png", large_image)
    cv2.imwrite(f"images/{option[4]}/{option[2]}-small.png", small_image)
    # show the output image
    # cv2.imshow('sea_image.jpg', im_v)
