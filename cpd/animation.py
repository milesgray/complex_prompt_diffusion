
import torch
from skimage.exposure import match_histograms
import cv2
from einops import rearrange, repeat

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)

def match_colors(x, s):
    prev_img_lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    color_match_lab = cv2.cvtColor(s, cv2.COLOR_RGB2LAB)
    matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

def lerp(x,y,a):
    return (1-a)*x+a*y
def sqrt_lerp(x, y,  a):
    return a*x+np.sqrt(1-a)*y
def noiser(shape, seed, target=None):
    #torch.manual_seed(seed)
    result = torch.randn(shape, device="cpu")
    if target is not None:
        result = torch.from_numpy(match_histograms(result.cpu().numpy(), target.cpu().numpy(), multichannel=True))
    return result
def add_noise(x: torch.Tensor, strength: float, sequence=None) -> torch.Tensor:
    if sequence is None:
        return x + noise_mgr.sample_sequence(sequence) * strength
        #return x + noiser(x.shape, opt.seed, sequence).cuda()
    else:
        return x + torch.randn(x.shape, device=x.device) * strength
    

def build_cycle_mod(n=5):
    return [x for x in range(1,n)] + [-x for x in range(1,n)][::-1]

def prepare_sample(x: torch.Tensor, coherance: float, diversity: float, target_noise=None, verbose: bool=False) -> torch.Tensor:
    if x is None:     
        #seed_everything(opt.seed, verbose=verbose)   
        x = torch.randn((1,4,96,64), device=device)
        #x = noise_mgr.sample_sequence("prepare_sample")
    else:
        if verbose: print(f"[prepare]\t[x_sample]\t{x.mean()}")
        x = sample_from_cv2(x).cuda()
        #seed_everything(opt.seed) #, verbose=verbose)
        x = add_noise(x, coherance, sequence=target_noise)
        x = x.half()
        
        if verbose: print(f"[prepare]\t[x_sample_noised]\t{x.mean()}")
        model_dict["vae"].half()
        with torch.no_grad():
            dist = model_dict["vae"].encode(x)
        if verbose: print(f"[prepare]\t[dist]\t{dist.parameters.mean()}")
        seed_everything(opt.seed) #, verbose=verbose)          
        x = dist.mean.cuda() + dist.std.cuda() * torch.randn(dist.mean.shape, device="cpu").cuda()
        #x = dist.mean.cuda() + dist.std.cuda() * noiser(dist.std.shape, opt.seed, target_noise).cuda()
        x = 0.18215 * x
        #seed_everything(opt.seed) #, verbose=verbose)  
        # if verbose: print(f"[prepare]\t[x_latent]\t{x.mean()}")
        x = sqrt_lerp(x, torch.randn(x.shape, device="cpu").cuda(), diversity)
    z_enc = x
    return z_enc

def lerp_latent(z: torch.Tensor, init_z: torch.Tensor, diversity: float, grounding: float) -> torch.Tensor:
    z = lerp(z, noise_mgr.sample_sequence("prepare_sample"), diversity)
    z = lerp(z, init_z, grounding)
    return z

def get_animation_step_params():
    params = {
        "prompt_start": prompt_start,
        "prompt_target": prompt_target,
        "prompt_fn": prompt_fn,
        "filters": [{
            "prompt": prompt_filter,
            "strength": lambda i, s: 0.4,
            "mask": None
        },
        {
            "prompt": "first person one point perspective",
            "strength": lambda i, s: 0.72,
            "mask": None
        },
        {
            "prompt": "ground floor",
            "strength": lambda i, s: 0.5,
            "mask": perspective_mask
        }],
        "lerps": [{
            "prompt": prompt_target,
            "args": {
                "magnitude": lambda i, s: 0.35 + (0.6 * i/s),
                "embed_k": lambda i, s: 100 + int(568 * i/s),
                "embed_range": lambda i, s: (0,768),
                "embed_largest": lambda i, s: True,
                "token_k": lambda i, s: None,
                "token_idxs": lambda i, s: [n for n in range(1,len(prompt_target.split(" ")))]
            }
        }],
        "W": opt.W,
        "H": opt.H,
        "seed": opt.seed,
        "anim_steps": anim_steps,
        "render_args": {
            "verbose": False,
            "do_clear": False,
            "do_display": False,
            "clip_guidance_freq": 3,
            "clip_guidance_scale": 0,
            "clip_guidance_grad_scale": 4,
        },
        "depth_args": depth_args,
    }
def to_args(d, idx, total_steps):
    return {k: v(idx, total_steps) for k,v in d.items()}
def render_animation_step(i, params, **kwargs):
    do_save = kwargs.get("do_save", False)
    do_clear = kwargs.get("do_clear", False)
    out_path = kwargs.get("out_path", Path())
    strength = kwargs.get("strength", 0.01)
    coherance = kwargs.get("coherance", 0.98)
    diversity = kwargs.get("diversity", 0.)
    steps = kwargs.get("steps", 10)
    prev_sample = kwargs.get("prev_sample", None)
    init_sample = kwargs.get("init_sample", None)
    prompt_start = params["prompt_start"]
    prompt_fn = params["prompt_fn"]

    cpe = prompt_fn(prompt_start)
    for args_dict in params["filters"]:
        cpe.add_filter(args_dict["prompt"], strength=args_dict["strength"](i, params["anim_steps"]), mask=args_dict["mask"])
    for args_dict in params["lerps"]:
        cpe.add_prompt_lerp(args_dict["prompt"], {k: v(i, params["anim_steps"]) for k,v 
                                        in args_dict["args"].items()})
    
    decode = False
    if prev_sample is not None:
        prev_sample = match_colors(prev_sample, init_sample)
        anim_args = get_anim_args(params["W"], params["H"], 
                                  img=prev_sample, 
                                  do_depth=True, 
                                  depth_args=params["depth_args"], 
                                  max_frames=params["anim_steps"])
        keys = KeyFrames(anim_args)
        prev_sample = do_3d_animation_step(prev_sample, i, keys, anim_args)
        prev_sample = do_2d_animation_step(prev_sample, i, keys, anim_args)
        latent = prepare_sample(prev_sample, coherance, diversity, target_noise=target_noise)
        #latent = sample_from_cv2(do_2d_animation_step(sample_to_cv2(latent), i, keys, anim_args))
        #latent = sqrt_lerp(latent, torch.randn(latent.shape, device=latent.device), diversity)
        
        decode = True
    else:
        seed_everything(params["seed"])
        latent = torch.randn((1,4,96,64), device=device)
    
    result = cpe.render(steps=steps,
                        latent=latent.cuda(),
                        prompt_txt=cpe.data,
                        denoising_strength=strength,
                        decode=decode,
                        **params["render_args"])
    img, sample = result[0], result[1]
    if do_clear:
        display.clear_output(wait=True)
    if do_save:
        save_image(Image.fromarray(img), out_path)
    display.display(Image.fromarray(img))    
    sample = sample_to_cv2(sample)
    return img, sample

def get_anim_args(w, h, img=None, max_frames=60, do_depth=False, depth_args=None):
    #parse_key_frames("10: (0.5), 50: (1.5), 60: (1)")
    args = Map()
    if do_depth:
        args.depth_map = get_depth(img, depth_args)
    args.max_frames = max_frames
    args.angle = "0: (0), 60: (0)"
    args.zoom = "0: (1.0), 60: (1.)"
    args.pan_x = "0: (0), 60: (0)"
    args.pan_y = "0: (0), 60: (0)"
    args.translation_x = "0: (0), 60: (0)"
    args.translation_y = "0: (0), 60: (0)"
    args.translation_z = "0: (.0), 60: (.0)"
    args.perspective_flip_theta =  "0: (0), 60: (0)"
    args.perspective_flip_phi =  "0: (1.04), 60: (1.05)"
    args.perspective_flip_gamma =  "0: (0), 60: (0)"
    args.perspective_flip_fv =  "0: (30), 60: (30)"
    args.rotation_3d_x = "0: (0), 30: (0) 60: (0)"
    args.rotation_3d_y = "0: (0), 60: (0)"
    args.rotation_3d_z = "0: (0), 60: (0)"
    args.coherance_schedule = "0: (0), 60: (0)"
    args.strength_schedule = "0: (0), 60: (0)"
    args.contrast_schedule = "0: (0), 60: (0)"

    args.near_plane = 200 # near_plane = gr.Slider(minimum=0, maximum=2000, step=1, label='Near Plane', value=200, visible=True)#near_plane
    args.far_plane = 1000 #gr.Slider(minimum=0, maximum=2000, step=1, label='Far Plane', value=1000, visible=True)#far_plane
    args.fov = 20 #gr.Slider(minimum=0, maximum=360, step=1, label='FOV', value=40, visible=True)#fov
    args.padding_mode = "reflection" #gr.Dropdown(label='Padding Mode', choices=['border', 'reflection', 'zeros'], value='border', visible=True)#padding_mode
    args.sampling_mode = "bicubic" #gr.Dropdown(label='Sampling Mode', choices=['bicubic', 'bilinear', 'nearest'], value='bicubic', visible=True)#sampling_mode
    args.border = "wrap"
    args.flip_2d_perspective = False

    args.w, args.h = w, h#get_width_height(args.img)

    return args    

class KeyFrames:
    def __init__(self, args):
        self.angle_series = build_key_frames(args.angle, 
                                            args.max_frames)
        self.zoom_series = build_key_frames(args.zoom, 
                                            args.max_frames)
        self.pan_x_series = build_key_frames(args.pan_x, 
                                            args.max_frames)
        self.pan_y_series = build_key_frames(args.pan_y, 
                                            args.max_frames)
        self.translation_x_series = build_key_frames(args.translation_x, 
                                                    args.max_frames)
        self.translation_y_series = build_key_frames(args.translation_y, 
                                                    args.max_frames)
        self.translation_z_series = build_key_frames(args.translation_z, 
                                                    args.max_frames)
        self.perspective_flip_theta_series = build_key_frames(args.perspective_flip_theta,
                                                              args.max_frames)
        self.perspective_flip_phi_series = build_key_frames(args.perspective_flip_phi,
                                                            args.max_frames)
        self.perspective_flip_gamma_series = build_key_frames(args.perspective_flip_gamma,
                                                              args.max_frames)
        self.perspective_flip_fv_series = build_key_frames(args.perspective_flip_fv,
                                                           args.max_frames)
        self.rotation_3d_x_series = build_key_frames(args.rotation_3d_x, 
                                                     args.max_frames)
        self.rotation_3d_y_series = build_key_frames(args.rotation_3d_y, 
                                                     args.max_frames)
        self.rotation_3d_z_series = build_key_frames(args.rotation_3d_z, 
                                                     args.max_frames)
        self.coherance_schedule_series = build_key_frames(args.coherance_schedule, 
                                                          args.max_frames)
        self.strength_schedule_series = build_key_frames(args.strength_schedule,
                                                         args.max_frames)
        self.contrast_schedule_series = build_key_frames(args.contrast_schedule,
                                                         args.max_frames)

def build_key_frames(raw, max_frames, prompt_parser=None, integer=False, interp_method='Linear'):
    frames = parse_key_frames(raw, prompt_parser=prompt_parser)
    return interpolate_key_frames(frames, max_frames, integer=integer, interp_method=interp_method)

def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames

def interpolate_key_frames(key_frames, max_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(),limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def do_3d_animation_step(img_np, frame_idx, keys, args):
    TRANSLATION_SCALE = 1.0/200.0 # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE,
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE,
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]),
        math.radians(keys.rotation_3d_y_series[frame_idx]),
        math.radians(keys.rotation_3d_z_series[frame_idx])
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    pixel_aspect = 1.0 # aspect of an individual pixel (so usually 1.0)

    near, far, fov_deg = args.near_plane, args.far_plane, args.fov
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, 
                                              fov=fov_deg, degrees=True, 
                                              device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, 
                                              fov=fov_deg, degrees=True, 
                                              R=rot_mat, 
                                              T=torch.tensor([translate_xyz]), 
                                              device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,args.h,dtype=torch.float32,device=device),
                         torch.linspace(-1.,1.,args.w,dtype=torch.float32,device=device))
    z = torch.as_tensor(args.depth_map, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,args.h,args.w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (args.h,args.w,2)).unsqueeze(0)

    image_tensor = torchvision.transforms.functional.to_tensor(Image.fromarray(img_np)).to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1/512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=args.sampling_mode,
        padding_mode=args.padding_mode,
        align_corners=False
    )

    # convert back to cv2 style numpy array 0->255 uint8
    result = rearrange(
        new_image.squeeze().clamp(0,1) * 255.0,
        'c h w -> h w c'
    ).cpu().numpy().astype(np.uint8)

    return result

def do_2d_animation_step(img_np, frame_idx, keys, args):
    angle = keys.angle_series[frame_idx]
    zoom = keys.zoom_series[frame_idx]
    pan_x = keys.pan_x_series[frame_idx]
    pan_y = keys.pan_y_series[frame_idx]

    center = (args.w // 2, args.h // 2)
    trans_mat = np.float32([[1, 0, pan_x], [0, 1, pan_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    if args.flip_2d_perspective:
        perspective_flip_theta = keys.perspective_flip_theta_series[frame_idx]
        perspective_flip_phi = keys.perspective_flip_phi_series[frame_idx]
        perspective_flip_gamma = keys.perspective_flip_gamma_series[frame_idx]
        perspective_flip_fv = keys.perspective_flip_fv_series[frame_idx]
        M, sl = warp_matrix(args.w, args.h, perspective_flip_theta, perspective_flip_phi, perspective_flip_gamma, 1.,
                           perspective_flip_fv);
        post_trans_mat = np.float32([[1, 0, (args.w - sl) / 2], [0, 1, (args.h - sl) / 2]])
        post_trans_mat = np.vstack([post_trans_mat, [0, 0, 1]])
        bM = np.matmul(M, post_trans_mat)
        xform = np.matmul(bM, rot_mat, trans_mat)
    else:
        xform = np.matmul(rot_mat, trans_mat)

    return cv2.warpPerspective(
        img_np,
        xform,
        (img_np.shape[1], img_np.shape[0]),
        borderMode=cv2.BORDER_WRAP if args.border == 'wrap' else cv2.BORDER_REPLICATE
    )

def construct_RotationMatrixHomogenous(rotation_angles):
    assert (type(rotation_angles) == list and len(rotation_angles) == 3)
    RH = np.eye(4, 4)
    cv2.Rodrigues(np.array(rotation_angles), RH[0:3, 0:3])
    return RH

# https://en.wikipedia.org/wiki/Rotation_matrix
def make_rotation_matrix(rotation_angles):
    rotation_angles = [np.deg2rad(x) for x in rotation_angles]

    phi = rotation_angles[0]  # around x
    gamma = rotation_angles[1]  # around y
    theta = rotation_angles[2]  # around z

    # X rotation
    Rphi = np.eye(4, 4)
    sp = np.sin(phi)
    cp = np.cos(phi)
    Rphi[1, 1] = cp
    Rphi[2, 2] = Rphi[1, 1]
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    # Y rotation
    Rgamma = np.eye(4, 4)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    Rgamma[0, 0] = cg
    Rgamma[2, 2] = Rgamma[0, 0]
    Rgamma[0, 2] = sg
    Rgamma[2, 0] = -sg

    # Z rotation (in-image-plane)
    Rtheta = np.eye(4, 4)
    st = np.sin(theta)
    ct = np.cos(theta)
    Rtheta[0, 0] = ct
    Rtheta[1, 1] = Rtheta[0, 0]
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    R = reduce(lambda x, y: np.matmul(x, y), [Rphi, Rgamma, Rtheta])

    return R

def perspective_tranform_args(ptsIn, ptsOut, W, H, sidelength):
    ptsIn2D = ptsIn[0, :]
    ptsOut2D = ptsOut[0, :]
    ptsOut2Dlist = []
    ptsIn2Dlist = []

    for i in range(0, 4):
        ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
        ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

    pin = np.array(ptsIn2Dlist) + [W / 2., H / 2.]
    pout = (np.array(ptsOut2Dlist) + [1., 1.]) * (0.5 * sidelength)
    pin = pin.astype(np.float32)
    pout = pout.astype(np.float32)

    return pin, pout

def warp_matrix(W, H, theta, phi, gamma, scale, fV):
    # M is to be estimated
    M = np.eye(4, 4)

    fVhalf = np.deg2rad(fV / 2.)
    d = np.sqrt(W * W + H * H)
    sideLength = scale * d / np.cos(fVhalf)
    h = d / (2.0 * np.sin(fVhalf))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    # Translation along Z-axis by -h
    T = np.eye(4, 4)
    T[2, 3] = -h

    # Rotation matrices around x,y,z
    R = make_rotation_matrix([phi, gamma, theta])

    # Projection Matrix
    P = np.eye(4, 4)
    P[0, 0] = 1.0 / np.tan(fVhalf)
    P[1, 1] = P[0, 0]
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    # pythonic matrix multiplication
    F = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

    # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
    ptsIn = np.array([[
        [-W / 2.,  H / 2., 0.], 
        [ W / 2.,  H / 2., 0.], 
        [ W / 2., -H / 2., 0.], 
        [-W / 2., -H / 2., 0.]
    ]])
    ptsOut = np.array(np.zeros((ptsIn.shape), dtype=ptsIn.dtype))
    ptsOut = cv2.perspectiveTransform(ptsIn, F)

    ptsInPt2f, ptsOutPt2f = perspective_tranform_args(ptsIn, ptsOut, W, H, sideLength)

    # check float32 otherwise OpenCV throws an error
    assert (ptsInPt2f.dtype == np.float32)
    assert (ptsOutPt2f.dtype == np.float32)
    M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)

    return M33, sideLength    