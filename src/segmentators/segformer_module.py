
import os
from collections import OrderedDict

import torch
from PIL import Image

from core import device as global_device

_MASK_CACHE_MAX = 128  # CPU seg maps kept in memory; at ~1 MB each this caps usage at ~128 MB


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
    Semantic-segmentation-based keypoint/match annotator using SegFormer.

    Runs SegFormer-B0 on Cityscapes labels to build a per-pixel keep/discard
    mask ('seg_mask', one per image), then, non-destructively, marks which
    keypoints ('keypt_mask') and matches (an extra column appended to
    'm_mask') fall on an excluded semantic class. Nothing is removed — all
    original keypoints/matches are preserved so the pre-filtering data stays
    available for later use.

    The dense per-pixel class map used to build 'seg_mask' (the full
    segmented-instances output, one class id per pixel) is heavier than a
    binary mask, so it isn't stored in the hdf5 dict: it's cached to files
    under `seg_cache_dir`, mirroring the image it came from, plus an
    in-process LRU cache for repeated lookups within a run.

    Two stages are supported, selected via `stage`:
    - 'keypoints' (default, single_image module): if 'kp' is present for the
      image, computes 'keypt_mask' — one bool per keypoint.
    - 'matches' (pair module, place after the matcher/geometric-verification
      modules in the pipeline): if 'm_idx'/'m_mask' are present, appends a
      segmentation column to 'm_mask', turning it from [M] into [M, 2]. Since
      this changes 'm_mask' from 1D to 2D, any module placed after this one
      that expects a 1D 'm_mask' must be updated accordingly — this module is
      meant to be the last one touching 'm_mask' in a pipeline.

    Attributes:
        exclude_classes: Cityscapes class names to discard. Defaults to sky,
            vehicles, and pedestrians — everything that hurts SfM of buildings
            and city squares.
        model_name: HuggingFace model id. Must be a SegFormer trained on
            Cityscapes (19-class label space).
        stage: 'keypoints' or 'matches' — see above.
        seg_cache_dir: Directory the dense per-pixel class maps are cached
            to, one file per (model_name, image).
    """

    # Shared across all instances: (model_name, device) -> (processor, model)
    _MODEL_CACHE: dict = {}
    # Shared across all instances: (model_name, img_path) -> CPU seg_map (argmax class ids)
    # LRU-capped so it never accumulates unbounded across a large image set.
    _SEGMAP_CACHE: OrderedDict = OrderedDict()

    def __init__(
        self,
        exclude_classes=None,
        model_name='nvidia/segformer-b0-finetuned-cityscapes-512-1024',
        stage='keypoints',
        seg_cache_dir='aux/seg_cache',
        **args,
    ):
        if stage not in ('keypoints', 'matches'):
            raise ValueError(f"stage must be 'keypoints' or 'matches', got {stage!r}")

        self.stage = stage
        self.single_image = stage == 'keypoints'
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
        self.seg_cache_dir = seg_cache_dir
        self._processor = None
        self._model = None

        classes_tag = '_'.join(c.replace(' ', '') for c in self.exclude_classes)
        stage_tag = 'matches_' if stage == 'matches' else ''
        id_more = args.get('id_more', '')
        self.id_string = f'segformer_{stage_tag}{classes_tag}'
        if id_more:
            self.id_string += f'_{id_more}'

    def _load_model(self):
        cache_key = (self.model_name, self.device)
        cached = segformer_module._MODEL_CACHE.get(cache_key)
        if cached is None:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
            processor = SegformerImageProcessor.from_pretrained(self.model_name)
            model = (
                SegformerForSemanticSegmentation.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )
            cached = (processor, model)
            segformer_module._MODEL_CACHE[cache_key] = cached

        self._processor, self._model = cached

    def _seg_cache_path(self, img_path: str) -> str:
        name = os.path.basename(img_path)
        return os.path.join(self.seg_cache_dir, self.model_name.replace('/', '_'), name + '.pt')

    def _get_seg_map(self, img_path: str, W: int, H: int) -> torch.Tensor:
        cache_key = (self.model_name, img_path)
        cache = segformer_module._SEGMAP_CACHE
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key].to(self.device)

        cache_path = self._seg_cache_path(img_path)
        if os.path.isfile(cache_path):
            seg_map = torch.load(cache_path, map_location='cpu')
        else:
            if self._model is None:
                self._load_model()

            image = Image.open(img_path).convert('RGB')
            inputs = self._processor(images=image, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits  # [1, C, h, w]

            seg_map = torch.nn.functional.interpolate(
                logits, size=(H, W), mode='bilinear', align_corners=False
            ).argmax(dim=1).squeeze(0).cpu().to(torch.uint8)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(seg_map, cache_path)

        # Store on CPU so the in-memory cache never ties up VRAM
        cache[cache_key] = seg_map
        if len(cache) > _MASK_CACHE_MAX:
            cache.popitem(last=False)

        return seg_map.to(self.device)

    def _get_keep_mask(self, img_path: str, W: int, H: int) -> torch.Tensor:
        seg_map = self._get_seg_map(img_path, W, H)

        keep_px = torch.ones((H, W), dtype=torch.bool, device=self.device)
        for class_id in self.exclude_ids:
            keep_px &= seg_map != class_id

        return keep_px

    def get_id(self):
        return self.id_string

    def finalize(self):
        segformer_module._SEGMAP_CACHE.clear()

    def run(self, **args):
        if self.stage == 'matches':
            return self._run_matches(**args)
        return self._run_keypoints(**args)

    def _run_keypoints(self, **args):
        idx = args['idx']
        img_path = args['img'][idx]

        W, H = Image.open(img_path).size
        keep_px = self._get_keep_mask(img_path, W, H)

        result = {'seg_mask': keep_px.cpu()}

        if 'kp' in args:
            kp = args['kp'][idx]           # [N, 2]  (x=col, y=row)
            xs = kp[:, 0].long().clamp(0, W - 1)
            ys = kp[:, 1].long().clamp(0, H - 1)
            result['keypt_mask'] = keep_px[ys, xs].cpu()  # [N] bool, non-destructive

        return result

    def _run_matches(self, **args):
        if 'm_idx' not in args:
            return {}

        img0_path, img1_path = args['img'][0], args['img'][1]
        kp0, kp1 = args['kp'][0], args['kp'][1]

        W0, H0 = Image.open(img0_path).size
        W1, H1 = Image.open(img1_path).size
        keep_px0 = self._get_keep_mask(img0_path, W0, H0)
        keep_px1 = self._get_keep_mask(img1_path, W1, H1)

        m_idx = args['m_idx']          # [M, 2] indices into kp0, kp1

        xs0 = kp0[m_idx[:, 0], 0].long().clamp(0, W0 - 1)
        ys0 = kp0[m_idx[:, 0], 1].long().clamp(0, H0 - 1)
        xs1 = kp1[m_idx[:, 1], 0].long().clamp(0, W1 - 1)
        ys1 = kp1[m_idx[:, 1], 1].long().clamp(0, H1 - 1)

        seg_keep = (keep_px0[ys0, xs0] & keep_px1[ys1, xs1]).to(args['m_mask'].device)  # [M] bool

        # Non-destructive: keep every match, append the segmentation verdict
        # as a second column instead of overwriting/removing rows, so the
        # pre-segmentation mask is still available.
        return {'m_mask': torch.stack([args['m_mask'], seg_keep], dim=1)}
