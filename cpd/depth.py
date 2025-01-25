import math
import cv2
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
try:
    from infer import InferenceHelper
    ADABINS_ENABLED = True
except:
    ADABINS_ENABLED = False
try:
    from midas.dpt_depth import DPTDepthModel
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    MIDAS_ENABLED = True
except:
    MIDAS_ENABLED = False
try:
    from pix2pix.options.test_options import TestOptions
    from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
    PIX2PIX_ENABLED = True
except:
    PIX2PIX_ENABLED = False

try:
    from lib.multi_depth_model_woauxi import RelDepthModel
    from lib.net_tools import load_ckpt
    LERES_ENABLED = True
except:
    LERES_ENABLED = False


def get_width_height(img):
    if img.shape[-1] == 3:
        w = img.shape[-2]
        h = img.shape[-3]
    elif img.shape[-3] == 3:
        w = img.shape[-1]
        h = img.shape[-2]
    else:
        raise ValueError("unknown image dimension format")
    return w, h

def load_midas(optimize=True):    
    midas_model = DPTDepthModel(
        path=f"/content/dpt_large-midas-2f21e586.pt",
        backbone="vitl16_384",
        non_negative=True,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    midas_transform = T.Compose([
        Resize(
            384, 384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="minimal",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        normalization,
        PrepareForNet()
    ])

    midas_model.eval()
    if optimize:    
        midas_model = midas_model.to(memory_format=torch.channels_last)
        midas_model = midas_model.half()
    midas_model = midas_model.half()
    midas_model = midas_model.cuda()

    return midas_model, midas_transform
def load_adabins():
    adabins_helper = InferenceHelper(dataset='nyu', device="cuda")
    return adabins_helper
def load_leres():
    depth_model = RelDepthModel(backbone='resnext101')
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

def apply_adabins(img, adabins_helper, device='cuda'):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    w, h = get_width_height(img)
    print(f"Estimating depth of {w}x{h} image with AdaBins...")
    MAX_ADABINS_AREA = 500000
    MIN_ADABINS_AREA = 448*448

    # resize image if too large or too small
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    image_pil_area = w*h
    resized = True
    if image_pil_area > MAX_ADABINS_AREA:
        scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
        depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is good for downsampling
        print(f"  resized to {depth_input.width}x{depth_input.height}")
    elif image_pil_area < MIN_ADABINS_AREA:
        scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
        depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        print(f"  resized to {depth_input.width}x{depth_input.height}")
    else:
        depth_input = img_pil
        resized = False

    # predict depth and resize back to original dimensions
    try:
        _, adabins_depth = adabins_helper.predict_pil(depth_input)
        if resized:
            adabins_depth = T.functional.resize(
                torch.from_numpy(adabins_depth),
                torch.Size([h, w]),
                interpolation=T.functional.InterpolationMode.BICUBIC
            )
        if isinstance(adabins_depth, np.ndarray):
            adabins_depth = torch.from_numpy(adabins_depth)
        adabins_depth = adabins_depth.squeeze()
    except:
        print(f"  exception encountered, falling back to pure MiDaS")
        use_adabins = False
    torch.cuda.empty_cache()
    return adabins_depth.to(device)

def apply_midas(img, midas_model, midas_transform, device='cuda'):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    w, h = get_width_height(img)
    # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
    img_midas = img.astype(np.float32) / 255.0
    img_midas_input = midas_transform({"image": img_midas})["image"]

    # MiDaS depth estimation implementation
    print(f"Estimating depth of {w}x{h} image with MiDaS...")
    sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
    sample = sample.to(memory_format=torch.channels_last)
    sample = sample.half()

    midas_depth = midas_model.forward(sample)
    midas_depth = torch.nn.functional.interpolate(
        midas_depth.unsqueeze(1),
        size=img_midas.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    midas_depth = midas_depth.cpu().detach().numpy()
    torch.cuda.empty_cache()

    # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
    midas_depth = np.subtract(50.0, midas_depth)
    midas_depth = midas_depth / 19.0

    depth_map = np.expand_dims(midas_depth, axis=0)
    depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)

    return depth_tensor

def apply_leres(img, model):
    
    A_resize = cv2.resize(img, (448, 448))
    rgb_half = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_LINEAR)

    if len(A_resize.shape) == 2:
        A_resize = A_resize[np.newaxis, :, :]
    if A_resize.shape[2] == 3:
        transform = T.Compose([T.ToTensor(),
                               T.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        A_resize = transform(A_resize)
    else:
        A_resize = A_resize.astype(np.float32)
        A_resize = torch.from_numpy(A_resize)
    img_torch = A_resize[None, :, :, :]

    pred_depth = model.inference(img_torch).cpu().numpy().squeeze()
    pred_depth_ori = cv2.resize(pred_depth, (img.shape[1], img.shape[0]))
    return pred_depth_ori

def get_depth_args(midas_weight=0.5):
    midas_model, midas_transform = load_midas()
    adabins_helper = load_adabins()
    return {
        "midas_model": midas_model,
        "midas_transform": midas_transform,
        "adabins_helper": adabins_helper,
        "midas_weight": midas_weight
    }
def get_depth(img, midas_weight=0.5):
    """Naive blending of MiDAS and ADABin depth models"""
    midas_model, midas_transform = load_midas()
    adabins_helper = load_adabins()
    midas_result = apply_midas(img, midas_model, midas_transform)
    adabin_result = apply_adabins(img, adabins_helper)
    depth_map = midas_result * midas_weight + adabin_result * (1-midas_weight)
    return depth_map

def do_depth(img):
    """MiDAS depth model"""
    midas_model, midas_transform = load_midas()
    result = apply_midas(img, midas_model, midas_transform)
    return result 

def create_depth_mask(map, size=(64,64)):
    """Convert depth map to an image mask of a size"""
    depth_min, depth_max = torch.amin(map, dim=[1, 2, 3], keepdim=True), \
                           torch.amax(map, dim=[1, 2, 3], keepdim=True)
    display_depth = (map - depth_min) / (depth_max - depth_min)
    depth_image = Image.fromarray(
        (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
    sized = torch.nn.functional.interpolate(
        map,
        size=size,
        mode="bicubic",
        align_corners=False,
    )
    depth_min, depth_max = torch.amin(sized, dim=[1, 2, 3], keepdim=True), \
                           torch.amax(sized, dim=[1, 2, 3],  keepdim=True)
    mask = 2. * (sized - depth_min) / (depth_max - depth_min) - 1.
    
    return mask

def build_depth_mask(img, q=0.35, size=None):
    """
    Get an image mask from the MiDAS depth map, with 1 being the closest.
    `q` (float): zero out this lowest quantile of the mask
    `size` (tuple int): size of output, default to 1/8 of image
    """
    depth_map = do_depth(img)
    size = size if size else (depth_map.shape[0]//8, depth_map.shape[1]//8)
    mask = 1 - create_depth_mask(depth_map.unsqueeze(0).unsqueeze(0), size=size)
    mask = rescale(shave(mask, q))
    return mask

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
def shave(x, q):
    v = torch.quantile(x.float(), q, -1, keepdim=True)
    x[x < v] = 0.0
    return x

class DepthManager:
    def __init__(self, q=0, size=(64,64), device='cuda'):        
        self.q = q
        self.size = size
        self.device

    def apply_depth(img):
        raise NotImplementedError

    def get(self, img):
        """
        Get an image mask from the MiDAS depth map, with 1 being the closest.
        `q` (float): zero out this lowest quantile of the mask
        `size` (tuple int): size of output, default to 1/8 of image
        """
        depth_map = self.apply_depth(img)
        size = size if size else (depth_map.shape[0]//8, depth_map.shape[1]//8)
        mask = 1 - self.create_depth_mask(depth_map.unsqueeze(0).unsqueeze(0))
        mask = self.rescale(self.shave(mask))
        return mask
    
    def create_depth_mask(self, depth_map):
        """Convert depth map to an image mask of a size"""
        depth_min, depth_max = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True), \
                            torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        display_depth = (depth_map - depth_min) / (depth_max - depth_min)
        depth_image = Image.fromarray(
            (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
        sized = torch.nn.functional.interpolate(
            depth_map,
            size=self.size,
            mode="bicubic",
            align_corners=False,
        )
        depth_min, depth_max = torch.amin(sized, dim=[1, 2, 3], keepdim=True), \
                            torch.amax(sized, dim=[1, 2, 3],  keepdim=True)
        mask = 2. * (sized - depth_min) / (depth_max - depth_min) - 1.
        
        return mask

    def rescale(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def shave(self, x):
        v = torch.quantile(x.float(), self.q, -1, keepdim=True)
        x[x < v] = 0.0
        return x
    
class MidasDepthManager(DepthManager):
    def __init__(self, q=0, size=(64,64), device='cuda'):
        self.model, self.transform = load_midas()
        super().__init__(q=q, size=size, device=device)

    def apply_depth(self, img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        w, h = get_width_height(img)
        # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
        img_midas = img.astype(np.float32) / 255.0
        img_midas_input = self.transform({"image": img_midas})["image"]

        # MiDaS depth estimation implementation
        print(f"Estimating depth of {w}x{h} image with MiDaS...")
        sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

        midas_depth = self.model.forward(sample)
        midas_depth = torch.nn.functional.interpolate(
            midas_depth.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        midas_depth = midas_depth.cpu().detach().numpy()
        torch.cuda.empty_cache()

        # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
        midas_depth = np.subtract(50.0, midas_depth)
        midas_depth = midas_depth / 19.0

        depth_map = np.expand_dims(midas_depth, axis=0)
        depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)

        return depth_tensor
    

def boost_depth(img1, img2, size, checkpoints_dir):
    opt = TestOptions().parse()    
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = checkpoints_dir +'/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()

    return global_merge(img1, img2, size, pix2pixmodel)

# Generate a double-input depth estimation
def global_merge(low_res, high_res, pix2pixsize, model):
    

    # Generate the low resolution estimation
    estimate1 = low_res
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate1.min()
    depth_max = estimate1.max()
    

    if depth_max - depth_min > np.finfo("float").eps:
        estimate1 = (estimate1 - depth_min) / (depth_max - depth_min)
    else:
        estimate1 = 0


    # Generate the high resolution estimation
    estimate2 = high_res
    
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)
    depth_min = estimate2.min()
    depth_max = estimate2.max()

    #print(depth_min,depth_max)

    if depth_max - depth_min > np.finfo("float").eps:
        estimate2 = (estimate2 - depth_min) / (depth_max - depth_min)
    else:
        estimate2 = 0

    # Inference on the merge model
    model.set_input(estimate1, estimate2)
    model.test()
    visuals = model.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


import gdown
from util import ImageandPatchs, generatemask, \
    getGF_fromintegral, calculateprocessingres, rgb2gray,\
    applyGridpatch

# MIDAS
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet

# PIX2PIX : MERGE NET
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
#
## Download model wieghts
# Mergenet model
os.system("mkdir -p ./pix2pix/checkpoints/mergemodel/")
url = "https://drive.google.com/u/0/uc?id=1cU2y-kMbt0Sf00Ns4CN2oO9qPJ8BensP&export=download"
output = "./pix2pix/checkpoints/mergemodel/"
gdown.download(url, output, quiet=False)

url = "https://drive.google.com/uc?id=1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC"
output = "./midas/"
gdown.download(url, output, quiet=False)

#
# select device
device = torch.device("cpu")
print("device: %s" % device)

print("nvidia:", torch.cuda.device_count())

whole_size_threshold = 3000  # R_max from the paper
GPU_threshold = 1600 - 32 # Limit for the GPU (NVIDIA RTX 2080), can be adjusted
scale_threshold = 3  # Allows up-scaling with a scale up to 3

opt = TestOptions().parse()
opt.gpu_ids = []
global pix2pixmodel
pix2pixmodel = Pix2Pix4DepthModel(opt)
pix2pixmodel.save_dir = './pix2pix/checkpoints/mergemodel'
pix2pixmodel.load_networks('latest')
pix2pixmodel.netG.to(device)
pix2pixmodel.device = device
pix2pixmodel.eval()

midas_model_path = "midas/model.pt"
global midasmodel
midasmodel = MidasNet(midas_model_path, non_negative=True)
midasmodel.to(device)
midasmodel.eval()


mask_org = generatemask((3000, 3000))


def estimatemidas(img, msize):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = T.Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction

# Generate a single-input depth estimation
def singleestimate(img, msize, net_type):
    if msize > GPU_threshold:
        print(" \t \t DEBUG| GPU THRESHOLD REACHED", msize, '--->', GPU_threshold)
        msize = GPU_threshold

    return estimatemidas(img, msize)


def doubleestimate(img, size1, size2, pix2pixsize, net_type):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
                torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


def adaptiveselection(integral_grad, patch_bound_list, gf, factor):
    patchlist = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32 / factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_fromintegral(integral_grad, bbox) / (bbox[2] * bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patchlist[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step / 2)
                bbox_test[1] = bbox_test[1] - int(search_step / 2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_fromintegral(integral_grad, bbox_test) / (bbox_test[2] * bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patchlist[str(count)]['rect'] = bbox
            patchlist[str(count)]['size'] = bbox[2]
            count = count + 1

    # Return selected patches
    return patchlist


def generatepatchs(img, base_size, factor):
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) + \
                 np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum() / len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size / 2))
    stride = int(round(blsize * 0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patchs ...")
    patch_bound_list = adaptiveselection(grad_integral_image, patch_bound_list, gf, factor)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patchset = sorted(patch_bound_list.items(), key=lambda x: getitem(x[1], 'size'), reverse=True)
    return patchset


def generatedepth(img, type="Final"):
    mask = mask_org.copy()
    print(type)
    if type == "Final" or type == "R20":
        r_threshold_value = 0.2
    elif type == "R0":
        r_threshold_value = 0
    else:
        return np.zeros_like(img), "Please select on of the Model Types"

    print(type,r_threshold_value)
    img = (img / 255.0).astype("float32")
    input_resolution = img.shape

    # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the
    # main paper and section B of the supplementary material.
    whole_image_optimal_size, patch_scale = calculateprocessingres(img, 384,
                                                                   r_threshold_value, scale_threshold,
                                                                   whole_size_threshold)

    print('\t wholeImage being processed in :', whole_image_optimal_size)

    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(img, 384, whole_image_optimal_size, 1024, 0)


    if type == "R0" or type == "R20":
        result = cv2.resize(whole_estimate, (input_resolution[1], input_resolution[0]),
                   interpolation=cv2.INTER_CUBIC)
        result = (result * 255).astype('uint8')
        result_colored = cv2.applyColorMap(result, cv2.COLORMAP_INFERNO)
        result_colored = cv2.cvtColor(result_colored, cv2.COLOR_RGB2BGR)

        return result_colored, "Completed"

    factor = max(min(1, 4 * patch_scale * whole_image_optimal_size / whole_size_threshold), 0.2)

    print('Adjust factor is:', 1 / factor)

    # Compute the target resolution.
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size * img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size * img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size

    img = cv2.resize(img, (round(b / factor), round(a / factor)), interpolation=cv2.INTER_CUBIC)
    print('Target resolution: ', img.shape)

    # Extract selected patches for local refinement
    base_size = 384 * 2
    patchset = generatepatchs(img, base_size, factor)

    # Computing a scale in case user prompted to generate the results as the same resolution of the input.
    # Notice that our method output resolution is independent of the input resolution and this parameter will only
    # enable a scaling operation during the local patch merge implementation to generate results with the same
    # resolution as the input.
    mergein_scale = input_resolution[0] / img.shape[0]

    imageandpatchs = ImageandPatchs("", "temp.png", patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1] * mergein_scale),
                                                         round(img.shape[0] * mergein_scale)),
                                        interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

    print('\t Resulted depthmap res will be :', whole_estimate_resized.shape[:2])
    print('patchs to process: ' + str(len(imageandpatchs)))

    # Enumerate through all patches, generate their estimations and refining the base estimate.
    for patch_ind in range(len(imageandpatchs)):
        # Get patch information
        patch = imageandpatchs[patch_ind]  # patch object
        patch_rgb = patch['patch_rgb']  # rgb patch
        patch_whole_estimate_base = patch['patch_whole_estimate_base']  # corresponding patch from base
        rect = patch['rect']  # patch size and location
        patch_id = patch['id']  # patch ID
        org_size = patch_whole_estimate_base.shape  # the original size from the unscaled input
        print('\t processing patch', patch_ind, '|', rect)
        # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
        # field size of the network for patches to accelerate the process.
        patch_estimation = doubleestimate(patch_rgb, 384, int(384*2),
                                          1024, 0)

        patch_estimation = cv2.resize(patch_estimation, (1024, 1024),
                                      interpolation=cv2.INTER_CUBIC)
        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (1024, 1024),
                                               interpolation=cv2.INTER_CUBIC)

        # Merging the patch estimation into the base estimate using our merge network:
        # We feed the patch estimation and the same region from the updated base estimate to the merge network
        # to generate the target estimate for the corresponding region.
        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()

        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped + 1) / 2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        mapped = prediction_mapped

        # We use a simple linear polynomial to make sure the result of the merge network would match the values of
        # base estimate
        p_coef = np.polyfit(mapped.reshape(-1), patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)).reshape(mapped.shape)

        merged = cv2.resize(merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
        # and resize it to our needed size while merging the patches.
        if mask.shape != org_size:
            mask = cv2.resize(mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        # Update the whole estimation:
        # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
        # blending at the boundaries of the patch region.
        tobemergedto[h1:h2, w1:w2] = np.multiply(tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

    result = (imageandpatchs.estimation_updated_image * 255).astype('uint8')
    result_colored = cv2.applyColorMap(result,cv2.COLORMAP_INFERNO)
    result_colored = cv2.cvtColor(result_colored,cv2.COLOR_RGB2BGR)
    return result_colored, "Completed"
