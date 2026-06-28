# ComfyUI-Krea2WashControl

Local ComfyUI helper nodes for Krea2 / Krea2Edit wash-style img2img workflows.

The goal is to make Krea2 easier to use as an image washing workflow: preserve the source aspect ratio, composition, pose, body proportions, and character silhouette while reducing low-denoise artifacts such as dirty skin texture, dark spots, color blotches, symbol/watermark copying, and sticker-like visual noise.

This plugin is independent from `ComfyUI-ConditioningKrea2Rebalance`. Updating the upstream Krea2 Rebalance extension will not overwrite these nodes.

## What It Helps With

- Keep the output size and aspect ratio close to the input image instead of forcing every image into one fixed resolution.
- Improve pose, body proportion, silhouette, and composition control in Krea2 wash workflows.
- Reduce skin dark spots, speckles, horizontal dirty lines, and blotchy texture at low denoise values.
- Pre-clean black, white, or custom-colored symbols, stickers, captions, emoji, and watermarks before Krea2 reads them as visual structure.
- Keep a softer reference-image control path when higher denoise improves skin but starts drifting pose or framing.

## Nodes

### Krea2 Reference Size

Keeps the workflow size based on the reference image. The default alignment multiple is `1`, meaning no resize or dimension alignment is applied. Set it to `8` or `16` only when you need VAE, depth, ControlNet, or other latent-friendly dimensions.

### Krea2 Skin-Safe Rebalance

Builds a softer Krea2 conditioning path that helps preserve composition, pose, and body proportions while reducing reference-latent inheritance of skin noise. Use it in the positive conditioning chain.

### Krea2 Colored Symbol / Watermark Cleaner

Detects high-contrast black, white, or selected-color strokes such as text, emoji, stickers, captions, watermarks, and colorful icons, then lightly repairs them with local image colors.

Recommended placement: before both `Krea2 Skin-Safe Rebalance` and `VAEEncode`, so Krea2 visual tokens do not treat these small marks as content that must be copied.

### Krea2 Latent Skin Cleaner

Lightly cleans skin-like regions before `VAEEncode`, reducing low-denoise inheritance of rough skin texture and small dark artifacts.

### Krea2 Skin Spot Cleaner

Detects small dark spots in skin-like regions and repairs them with local color averaging. Useful for dark spots, speckles, dot-like spread, and mild dirty line artifacts.

## Common Parameters

- `reference_strength`: Higher values preserve the source pose and proportions more strongly, but can also carry over unwanted artifacts.
- `text_end_percent` / `image_start_percent`: Controls when text and image-reference conditioning are active during sampling.
- `align_multiple`: `1` means keep the original size; `8` or `16` aligns width and height to that multiple.
- `color_mode`: Choose `black_white`, white only, black only, custom colors, or black/white plus custom colors.
- `custom_colors`: Supports `#ffffff,#000000`, `#f07ac8`, and names such as `black`, `white`, `pink`, `orange`.
- `color_tolerance`: Higher values match a wider range around custom colors, but can also catch similar-looking details.
- `white_threshold`: Lower values catch more light symbols; black mode uses the opposite threshold for dark strokes.
- `contrast_threshold`: Lower values catch thinner strokes, but can also affect lace, clothing edges, or fine texture.
- `skin_smooth`: Higher values clean skin more, but can look soft or blurred.
- `spot_threshold`: Lower values catch more tiny dark spots, but can also affect shadows and clothing folds.
- `global_skin_smooth`: A light skin-only smoothing pass to reduce remaining dirty texture.

## Recommended Chain

```text
Reference image
-> Krea2 Reference Size
-> Krea2 Colored Symbol / Watermark Cleaner
-> Krea2 Latent Skin Cleaner
-> Krea2 Skin Spot Cleaner
-> VAEEncode
-> KSampler
```

Also connect the cleaned image from `Krea2 Colored Symbol / Watermark Cleaner` to `Krea2 Skin-Safe Rebalance`. Do not feed a source image with obvious watermarks, captions, stickers, or symbols directly into Krea2 visual reference conditioning, because the model may copy or amplify them.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/Z-yaofang/ComfyUI-Krea2WashControl.git
```

Restart ComfyUI after installation or updates.

## Notes

- This plugin does not include the Krea2 model itself.
- The current node UI labels are localized for the author's local Chinese ComfyUI setup.
- If ComfyUI still shows old node parameters after an update, restart ComfyUI and force-refresh the browser with `Ctrl+F5`.
- The cleaners are preprocessing helpers for Krea2 wash workflows. They are not full face restoration, inpainting, or general-purpose retouching tools.
