import math

import torch
import torch.nn.functional as F

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _split_bands(t, n_bands=12):
    flat = t.shape[-1]
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    if ref_dir.shape[0] == 1 and target_batch != 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    if ref_dir.shape[0] != target_batch:
        ref_dir = ref_dir.mean(dim=0, keepdim=True).expand(target_batch, *ref_dir.shape[1:])
    return ref_dir


def _parse_floats(s):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals


def _kw(kwargs, chinese_name, english_name=None, default=None):
    if chinese_name in kwargs:
        return kwargs[chinese_name]
    if english_name is not None and english_name in kwargs:
        return kwargs[english_name]
    return default


SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def compile_edit(
    clip,
    vae,
    prompt,
    image,
    strength=1.0,
    use_vision_tokens=True,
    use_reference_latents=True,
):
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Krea 2 Wash Control requires ComfyUI.")

    images_vl = None
    combined_latents = None
    if image is not None and (use_vision_tokens or use_reference_latents):
        samples = image.movedim(-1, 1)

        if use_vision_tokens:
            pixel_budget = 1_843_200
            src_pixels = samples.shape[3] * samples.shape[2]
            if src_pixels > pixel_budget:
                scale_by = math.sqrt(pixel_budget / src_pixels)
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            else:
                s = samples
            images_vl = [s.movedim(1, -1)]

            image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
            full_prompt = image_prompt + prompt
        else:
            full_prompt = prompt

        if use_reference_latents:
            latent = vae.encode(samples.movedim(1, -1)[:, :, :, :3])
            combined_latents = [latent * strength]
    else:
        full_prompt = prompt

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl,
        llama_template=SYS_TEMPLATE,
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    if combined_latents is not None:
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": combined_latents},
            append=True,
        )

    return conditioning


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=12):
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(
            cond_bands, ref_bands, n_bands, strength, per_band_strengths, sign=+1,
        )
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def dissim_guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(dissim_guidance_conditioning(item, ref_item, strength, per_band_strengths))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths)
        return structure
    return structure


def guidance(conditioning, reference, strength):
    return dissim_guidance_conditioning(conditioning, reference, strength, per_band_strengths=None)


def _soft_between(x, low, high, softness):
    softness = max(float(softness), 1e-4)
    return torch.sigmoid((x - low) / softness) * torch.sigmoid((high - x) / softness)


def _gaussian_blur_nhwc(image, radius):
    radius = int(radius)
    if radius <= 0:
        return image

    orig_dtype = image.dtype
    x = image.float().movedim(-1, 1)
    channels = x.shape[1]
    sigma = max(radius * 0.5, 0.5)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).to(dtype=x.dtype)

    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=channels)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky, groups=channels)
    return x.movedim(1, -1).to(dtype=orig_dtype)


def _skin_likeness_mask(image):
    img = image.float().clamp(0.0, 1.0)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    mask = _soft_between(cb, 0.25, 0.53, 0.035)
    mask = mask * _soft_between(cr, 0.47, 0.78, 0.035)
    mask = mask * _soft_between(y, 0.16, 0.94, 0.045)
    mask = mask * torch.sigmoid((r - b - 0.015) / 0.035)
    mask = mask * torch.sigmoid((r - g + 0.09) / 0.055)
    return mask.clamp(0.0, 1.0)


class Krea2WashRebalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "reference_strength": ("FLOAT", {"default": 1.35, "min": 0.0, "max": 3.0, "step": 0.05}),
            "text_end_percent": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
            "image_start_percent": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            "rebalance_multiplier": ("FLOAT", {"default": 4.50, "min": 0.0, "max": 12.0, "step": 0.05}),
            "guidance_strength": ("FLOAT", {"default": 0.35, "min": -3.0, "max": 3.0, "step": 0.05}),
            "main_layer_weights": ("STRING", {
                "default": "0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.4,10.0,1.2,1.4,1.0",
                "multiline": False,
            }),
            "subject_prefix": ("STRING", {
                "default": "(Subject:2.2) (same character:1.4) (same full-body pose:1.5) (same body silhouette and proportions:1.5) ",
                "multiline": True,
                "dynamicPrompts": True,
            }),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning/Krea2洗图"

    @staticmethod
    def _process_cond(cond, multiplier, main_layer_weights, guidance_strength):
        cond_ref = refocus(
            cond, multiplier,
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
        )
        cond_main = refocus(cond, multiplier, main_layer_weights)
        return guidance(cond_main, cond_ref, guidance_strength)

    def _main_impl(
        self,
        text,
        clip,
        vae,
        image,
        reference_strength=1.35,
        text_end_percent=0.08,
        image_start_percent=0.02,
        rebalance_multiplier=4.5,
        guidance_strength=0.35,
        main_layer_weights="0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.4,10.0,1.2,1.4,1.0",
        subject_prefix="(Subject:2.2) (same character:1.4) (same full-body pose:1.5) (same body silhouette and proportions:1.5) ",
        use_vision_tokens=True,
        use_internal_reference_latents=True,
    ):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Wash Control requires ComfyUI.")

        text_end_percent = max(0.0, min(1.0, text_end_percent))
        image_start_percent = max(0.0, min(1.0, image_start_percent))
        if text_end_percent < image_start_percent:
            text_end_percent = image_start_percent

        prompt = "{}{}".format(subject_prefix, text)

        cond_text = compile_edit(clip, vae, prompt, None, 1.0)
        cond_text = self._process_cond(
            cond_text, rebalance_multiplier, main_layer_weights, guidance_strength,
        )
        cond_text = node_helpers.conditioning_set_values(
            cond_text, {"start_percent": 0.000, "end_percent": text_end_percent},
        )

        cond_image = compile_edit(
            clip,
            vae,
            prompt,
            image,
            reference_strength,
            use_vision_tokens=use_vision_tokens,
            use_reference_latents=use_internal_reference_latents,
        )
        cond_image = self._process_cond(
            cond_image, rebalance_multiplier, main_layer_weights, guidance_strength,
        )
        cond_image = node_helpers.conditioning_set_values(
            cond_image, {"start_percent": image_start_percent, "end_percent": 1.000},
        )

        return (cond_text + cond_image,)

    def main(
        self,
        text,
        clip,
        vae,
        image,
        reference_strength=1.35,
        text_end_percent=0.08,
        image_start_percent=0.02,
        rebalance_multiplier=4.5,
        guidance_strength=0.35,
        main_layer_weights="0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.4,10.0,1.2,1.4,1.0",
        subject_prefix="(Subject:2.2) (same character:1.4) (same full-body pose:1.5) (same body silhouette and proportions:1.5) ",
    ):
        return self._main_impl(
            text,
            clip,
            vae,
            image,
            reference_strength,
            text_end_percent,
            image_start_percent,
            rebalance_multiplier,
            guidance_strength,
            main_layer_weights,
            subject_prefix,
            use_vision_tokens=True,
            use_internal_reference_latents=True,
        )


class Krea2WashRebalanceSkinSafe(Krea2WashRebalance):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "提示词": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "参考图": ("IMAGE",),
            "参考强度": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 3.0, "step": 0.05}),
            "文字结束进度": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
            "图像开始进度": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
            "重平衡倍率": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 12.0, "step": 0.05}),
            "差异引导强度": ("FLOAT", {"default": 0.08, "min": -3.0, "max": 3.0, "step": 0.05}),
            "主层权重": ("STRING", {
                "default": "0.0,0.8,0.0,0.0,0.0,0.0,0.0,0.8,2.0,0.8,0.8,0.8",
                "multiline": False,
            }),
            "主体前缀": ("STRING", {
                "default": "(Subject:1.45) (same composition:1.2) (similar pose and body proportions:1.15) (clean even natural skin tone:1.35) ",
                "multiline": True,
                "dynamicPrompts": True,
            }),
            "使用视觉Token": ("BOOLEAN", {"default": True}),
            "使用内部参考Latent": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning/Krea2洗图"

    def main(self, **kwargs):
        text = _kw(kwargs, "提示词", "text", "")
        clip = _kw(kwargs, "clip")
        vae = _kw(kwargs, "vae")
        image = _kw(kwargs, "参考图", "image")
        reference_strength = _kw(kwargs, "参考强度", "reference_strength", 0.55)
        text_end_percent = _kw(kwargs, "文字结束进度", "text_end_percent", 0.12)
        image_start_percent = _kw(kwargs, "图像开始进度", "image_start_percent", 0.18)
        rebalance_multiplier = _kw(kwargs, "重平衡倍率", "rebalance_multiplier", 2.0)
        guidance_strength = _kw(kwargs, "差异引导强度", "guidance_strength", 0.08)
        main_layer_weights = _kw(
            kwargs,
            "主层权重",
            "main_layer_weights",
            "0.0,0.8,0.0,0.0,0.0,0.0,0.0,0.8,2.0,0.8,0.8,0.8",
        )
        subject_prefix = _kw(
            kwargs,
            "主体前缀",
            "subject_prefix",
            "(Subject:1.45) (same composition:1.2) (similar pose and body proportions:1.15) (clean even natural skin tone:1.35) ",
        )
        use_vision_tokens = _kw(kwargs, "使用视觉Token", "use_vision_tokens", True)
        use_internal_reference_latents = _kw(
            kwargs,
            "使用内部参考Latent",
            "use_internal_reference_latents",
            False,
        )
        return self._main_impl(
            text,
            clip,
            vae,
            image,
            reference_strength,
            text_end_percent,
            image_start_percent,
            rebalance_multiplier,
            guidance_strength,
            main_layer_weights,
            subject_prefix,
            use_vision_tokens=use_vision_tokens,
            use_internal_reference_latents=use_internal_reference_latents,
        )


class Krea2WashReferenceSize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    align_modes = ["nearest", "down", "up"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "参考图": ("IMAGE",),
            "缩放算法": (cls.upscale_methods, {"default": "lanczos"}),
            "尺寸对齐倍数": ("INT", {"default": 8, "min": 8, "max": 128, "step": 8}),
            "对齐方式": (cls.align_modes, {"default": "nearest"}),
            "最大百万像素": ("FLOAT", {
                "default": 0.0,
                "min": 0.0,
                "max": 64.0,
                "step": 0.1,
                "tooltip": "0 表示不限制原图尺寸，只做倍数对齐。",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("对齐图像", "宽度", "高度")
    FUNCTION = "main"
    CATEGORY = "conditioning/Krea2洗图"

    @staticmethod
    def _align(value, multiple, mode):
        if multiple <= 1:
            return int(value)
        if mode == "down":
            return max(multiple, int(value // multiple) * multiple)
        if mode == "up":
            return max(multiple, int(math.ceil(value / multiple)) * multiple)
        return max(multiple, int(round(value / multiple)) * multiple)

    def main(self, **kwargs):
        image = _kw(kwargs, "参考图", "image")
        upscale_method = _kw(kwargs, "缩放算法", "upscale_method", "lanczos")
        align_multiple = _kw(kwargs, "尺寸对齐倍数", "align_multiple", 8)
        align_mode = _kw(kwargs, "对齐方式", "align_mode", "nearest")
        max_megapixels = _kw(kwargs, "最大百万像素", "max_megapixels", 0.0)

        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Wash Control requires ComfyUI.")

        height = int(image.shape[1])
        width = int(image.shape[2])

        if max_megapixels and max_megapixels > 0:
            max_pixels = float(max_megapixels) * 1024 * 1024
            pixels = width * height
            if pixels > max_pixels:
                scale_by = math.sqrt(max_pixels / pixels)
                width = max(align_multiple, round(width * scale_by))
                height = max(align_multiple, round(height * scale_by))

        width = self._align(width, align_multiple, align_mode)
        height = self._align(height, align_multiple, align_mode)

        if width == int(image.shape[2]) and height == int(image.shape[1]):
            return (image, width, height)

        samples = image.movedim(-1, 1)
        resized = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled").movedim(1, -1)
        return (resized, width, height)


class Krea2WashLatentSkinCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "图像": ("IMAGE",),
            "肤质平滑": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            "肤色遮罩扩张": ("INT", {"default": 5, "min": 0, "max": 31, "step": 1}),
            "细节保留": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            "主模糊半径": ("INT", {"default": 7, "min": 1, "max": 31, "step": 2}),
            "低通半径": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2}),
            "全局轻降噪": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("清理图像", "遮罩预览", "肤色遮罩")
    FUNCTION = "main"
    CATEGORY = "conditioning/Krea2洗图"

    def main(self, **kwargs):
        image = _kw(kwargs, "图像", "image")
        skin_smooth = _kw(kwargs, "肤质平滑", "skin_smooth", 0.65)
        skin_mask_expand = _kw(kwargs, "肤色遮罩扩张", "skin_mask_expand", 5)
        detail_keep = _kw(kwargs, "细节保留", "detail_keep", 0.35)
        blur_radius = _kw(kwargs, "主模糊半径", "blur_radius", 7)
        lowpass_radius = _kw(kwargs, "低通半径", "lowpass_radius", 3)
        global_denoise = _kw(kwargs, "全局轻降噪", "global_denoise", 0.10)

        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Wash Control requires ComfyUI.")

        img = image.float().clamp(0.0, 1.0)
        smooth = _gaussian_blur_nhwc(img, blur_radius)
        lowpass = _gaussian_blur_nhwc(img, lowpass_radius)
        detail_restored = smooth + (img - lowpass) * float(detail_keep)
        detail_restored = detail_restored.clamp(0.0, 1.0)

        mask = _skin_likeness_mask(img)
        if skin_mask_expand > 0:
            r = int(skin_mask_expand)
            m = mask.unsqueeze(1)
            m = F.pad(m, (r, r, r, r), mode="reflect")
            m = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1)
            mask = m.squeeze(1)
            mask = _gaussian_blur_nhwc(mask.unsqueeze(-1), max(1, r // 2)).squeeze(-1)

        mask = mask.clamp(0.0, 1.0) * float(skin_smooth)
        mask = mask.unsqueeze(-1)
        cleaned = img * (1.0 - mask) + detail_restored * mask

        if global_denoise > 0.0:
            global_soft = _gaussian_blur_nhwc(cleaned, 1)
            cleaned = cleaned * (1.0 - float(global_denoise)) + global_soft * float(global_denoise)

        mask_preview = torch.cat([
            mask.expand_as(img)[..., :1],
            torch.zeros_like(img[..., :1]),
            1.0 - mask.expand_as(img)[..., :1],
        ], dim=-1)
        return (cleaned.clamp(0.0, 1.0).to(image.dtype), mask_preview.to(image.dtype), mask.squeeze(-1))


class Krea2WashSkinSpotCleaner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "图像": ("IMAGE",),
            "清理强度": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
            "暗点阈值": ("FLOAT", {"default": 0.045, "min": 0.005, "max": 0.20, "step": 0.005}),
            "暗点扩张": ("INT", {"default": 2, "min": 0, "max": 12, "step": 1}),
            "修补半径": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2}),
            "肤色遮罩扩张": ("INT", {"default": 3, "min": 0, "max": 21, "step": 1}),
            "全局肤色平滑": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("清理图像", "暗点预览", "暗点遮罩")
    FUNCTION = "main"
    CATEGORY = "conditioning/Krea2洗图"

    def main(self, **kwargs):
        image = _kw(kwargs, "图像", "image")
        clean_strength = _kw(kwargs, "清理强度", "clean_strength", 0.85)
        spot_threshold = _kw(kwargs, "暗点阈值", "spot_threshold", 0.045)
        spot_expand = _kw(kwargs, "暗点扩张", "spot_expand", 2)
        patch_radius = _kw(kwargs, "修补半径", "patch_radius", 5)
        skin_mask_expand = _kw(kwargs, "肤色遮罩扩张", "skin_mask_expand", 3)
        global_skin_smooth = _kw(kwargs, "全局肤色平滑", "global_skin_smooth", 0.08)

        if not _COMFY_AVAILABLE:
            raise RuntimeError("Krea 2 Wash Control requires ComfyUI.")

        img = image.float().clamp(0.0, 1.0)
        skin = _skin_likeness_mask(img)
        if skin_mask_expand > 0:
            r = int(skin_mask_expand)
            m = skin.unsqueeze(1)
            m = F.pad(m, (r, r, r, r), mode="reflect")
            m = F.max_pool2d(m, kernel_size=2 * r + 1, stride=1)
            skin = _gaussian_blur_nhwc(m.squeeze(1).unsqueeze(-1), max(1, r // 2)).squeeze(-1)

        y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        local = _gaussian_blur_nhwc(img, patch_radius)
        local_y = 0.299 * local[..., 0] + 0.587 * local[..., 1] + 0.114 * local[..., 2]
        dark_residual = (local_y - y).clamp(0.0, 1.0)

        softness = max(float(spot_threshold) * 0.45, 0.005)
        spot = torch.sigmoid((dark_residual - float(spot_threshold)) / softness)
        spot = spot * skin.clamp(0.0, 1.0)

        if spot_expand > 0:
            r = int(spot_expand)
            s = spot.unsqueeze(1)
            s = F.pad(s, (r, r, r, r), mode="reflect")
            s = F.max_pool2d(s, kernel_size=2 * r + 1, stride=1)
            spot = s.squeeze(1)
            spot = _gaussian_blur_nhwc(spot.unsqueeze(-1), max(1, r)).squeeze(-1)

        spot = (spot * float(clean_strength)).clamp(0.0, 1.0)
        cleaned = img * (1.0 - spot.unsqueeze(-1)) + local * spot.unsqueeze(-1)

        if global_skin_smooth > 0.0:
            soft = _gaussian_blur_nhwc(cleaned, 2)
            skin_blend = (skin * float(global_skin_smooth)).clamp(0.0, 1.0).unsqueeze(-1)
            cleaned = cleaned * (1.0 - skin_blend) + soft * skin_blend

        preview = torch.stack([
            spot.clamp(0.0, 1.0),
            torch.zeros_like(spot),
            skin.clamp(0.0, 1.0) * 0.5,
        ], dim=-1)
        return (cleaned.clamp(0.0, 1.0).to(image.dtype), preview.to(image.dtype), spot)


NODE_CLASS_MAPPINGS = {
    "Krea2WashRebalanceSkinSafe": Krea2WashRebalanceSkinSafe,
    "Krea2WashReferenceSize": Krea2WashReferenceSize,
    "Krea2WashLatentSkinCleaner": Krea2WashLatentSkinCleaner,
    "Krea2WashSkinSpotCleaner": Krea2WashSkinSpotCleaner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2WashRebalanceSkinSafe": "Krea2洗图重平衡（肤质安全）",
    "Krea2WashReferenceSize": "Krea2洗图参考尺寸",
    "Krea2WashLatentSkinCleaner": "Krea2洗图Latent肤质清理",
    "Krea2WashSkinSpotCleaner": "Krea2洗图皮肤暗点清理",
}
