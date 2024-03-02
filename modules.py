from skimage import segmentation
from PIL import Image
from numpy import asarray
def generate_superpixels(image, scale, sigma):
    PIL_Image = Image.open(image)
    img = asarray(PIL_Image)
    seg_map = segmentation.felzenszwalb(img, scale=scale, sigma=sigma)
    boundaries = segmentation.mark_boundaries(img, seg_map)
    return boundaries