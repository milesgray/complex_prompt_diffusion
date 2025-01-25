
import numpy as np
from IPython import display
from PIL import Image


def render_callback(x0, i, **kwargs):
    x0 = x0.cpu().squeeze() \
                    .add(1) \
                    .div(2) \
                    .mul(255) \
                    .permute(1,2,0) \
                    .numpy() \
                        .astype(np.uint8)
                        
    im = Image.fromarray(x0)
    do_clear = kwargs.get("do_clear", False)
    if do_clear:
        display.clear_output(wait=True)
    do_display = kwargs.get("do_display", False)
    if do_display:
        display.display(im)