import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape
    
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    
    plt.show()