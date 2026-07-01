
from collections import OrderedDict

import torch
from PIL import Image

from core import device as global_device

_MASK_CACHE_MAX = 128  # CPU bool masks; at ~1 MB each this caps usage at ~128 MB


CITYSCAPES_LABEL2ID = {
    'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4,
    'pole': 5, 'traffic light': 6, 'traffic sign': 7, 'vegetation': 8,
    'terrain': 9, 'sky': 10, 'person': 11, 'rider': 12, 'car': 13,
    'truck': 14, 'bus': 15, 'train': 16, 'motorcycle': 17, 'bicycle': 18,
}

DEFAULT_EXCLUDE = [
    'sky', 'person', 'rider',
    'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


class segformer_module:
    """
    Post-detection keypoint filter using SegFormer semantic segmentation.

    Runs SegFormer-B0 on Cityscapes labels to build a per-pixel keep/discard
    mask, then drops any keypoints (and their kH, kr, desc) that land on
    excluded semantic classes. Designed to remove dynamic objects and sky
    before matching, improving 3-D reconstruction of static scenes.

    The segmentation mask is cached in memory per image path so that when
    the same module instance appears in multiple sub-pipelines (e.g. inside
    a pipeline_muxer_module), SegFormer inference runs only once per image.

    Attributes:
        exclude_classes: Cityscapes class names to discard. Defaults to sky,
            vehicles, and pedestrians — everything that hurts SfM of buildings
            and city squares.
        model_name: HuggingFace model id. Must be a SegFormer trained on
            Cityscapes (19-class label space).
    """

    def __init__(
        self,
        exclude_classes=None,
        model_name='nvidia/segformer-b0-finetuned-cityscapes-512-1024',
        **args,
    ):
        self.single_image = True
        self.pipeliner = False
        self.pass_through = False
        self.add_to_cache = True

        if 'add_to_cache' in args:
            self.add_to_cache = args['add_to_cache']

        self.device = torch.device(global_device)
        if 'device' in args:
            self.device = torch.device(args['device'])

        if exclude_classes is None:
            exclude_classes = list(DEFAULT_EXCLUDE)

        self.exclude_classes = sorted(set(exclude_classes))
        self.exclude_ids = frozenset(
            CITYSCAPES_LABEL2ID[c] for c in self.exclude_classes if c in CITYSCAPES_LABEL2ID
        )
        self.model_name = model_name

        classes_tag = '_'.join(c.replace(' ', '') for c in self.exclude_classes)
        id_more = args.get('id_more', '')
        self.id_string = f'segformer_{classes_tag}'
        if id_more:
            self.id_string += f'_{id_more}'

        self._processor = None
        self._model = None
        # img_path -> CPU bool mask; LRU-capped so VRAM never accumulates
        self._mask_cache: OrderedDict = OrderedDict()

    def _load_model(self):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self._processor = SegformerImageProcessor.from_pretrained(self.model_name)
        self._model = (
            SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )

    def _get_keep_mask(self, img_path: str, W: int, H: int) -> torch.Tensor:
        if img_path in self._mask_cache:
            self._mask_cache.move_to_end(img_path)
            return self._mask_cache[img_path].to(self.device)

        if self._model is None:
            self._load_model()

        image = Image.open(img_path).convert('RGB')
        inputs = self._processor(images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits  # [1, C, h, w]

        seg_map = torch.nn.functional.interpolate(
            logits, size=(H, W), mode='bilinear', align_corners=False
        ).argmax(dim=1).squeeze(0)

        keep_px = torch.ones((H, W), dtype=torch.bool, device=self.device)
        for class_id in self.exclude_ids:
            keep_px &= seg_map != class_id

        # Store on CPU so the cache never ties up VRAM
        self._mask_cache[img_path] = keep_px.cpu()
        if len(self._mask_cache) > _MASK_CACHE_MAX:
            self._mask_cache.popitem(last=False)

        return keep_px  # already on self.device from the computation above

    def get_id(self):
        return self.id_string

    def finalize(self):
        self._mask_cache.clear()

    def run(self, **args):
        idx = args['idx']
        img_path = args['img'][idx]

        W, H = Image.open(img_path).size
        keep_px = self._get_keep_mask(img_path, W, H)

        kp = args['kp'][idx]           # [N, 2]  (x=col, y=row)
        xs = kp[:, 0].long().clamp(0, W - 1)
        ys = kp[:, 1].long().clamp(0, H - 1)
        keep = keep_px[ys, xs]         # [N] bool

        result = {
            'kp': kp[keep],
            'kH': args['kH'][idx][keep],
            'kr': args['kr'][idx][keep],
        }

        if 'desc' in args:
            result['desc'] = args['desc'][idx][keep]

        return result
