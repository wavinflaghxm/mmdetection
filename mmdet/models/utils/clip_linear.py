import json
import clip
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob
from mmdet.models.utils.builder import LINEAR_LAYERS


def load_clip_features(
        type: str,
        ann_file: str,
        prompt: Optional[str] = "a",
        use_wordnet: bool = False,
        fix_space: bool = False,
        use_underscore: bool = False,
        avg_synonyms: bool = False,
        save_path: Optional[str] = None,
        **kwargs
):
    data = json.load(open(ann_file, "r"))
    cat_names = [x["name"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    if "synonyms" in data["categories"][0]:
        if use_wordnet:
            from nltk.corpus import wordnet
            synonyms = [[xx.name() for xx in wordnet.synset(x["synset"]).lemmas()] if x["synset"] != "stop_sign.n.01"
                        else ["stop_sign"] for x in sorted(data["categories"], key=lambda x: x["id"])]
        else:
            synonyms = [x["synonyms"] for x in sorted(data["categories"], key=lambda x: x["id"])]
    else:
        synonyms = []

    if fix_space:
        cat_names = [x.replace("_", " ") for x in cat_names]

    if use_underscore:
        cat_names = [x.strip().replace("/ ", "/").replace(" ", "_") for x in cat_names]

    if prompt == "a":
        sentences = ["a " + x for x in cat_names]
        sentences_synonyms = [["a " + xx for xx in x] for x in synonyms]
    elif prompt == "none":
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif prompt == "photo":
        sentences = ["a photo of a {}".format(x) for x in cat_names]
        sentences_synonyms = [["a photo of a {}".format(xx) for xx in x] for x in synonyms]
    elif prompt == "scene":
        sentences = ["a photo of a {} in the scene".format(x) for x in cat_names]
        sentences_synonyms = [["a photo of a {} in the scene".format(xx) for xx in x] for x in synonyms]
    elif prompt == "all":
        from mmdet.datasets.dataset_class_info import template_list
        sentences, sentences_synonyms = [], []
        for template in template_list:
            sentences.extend([template.format(x) for x in cat_names])
            sentences_synonyms.extend([[template.format(xx) for xx in x] for x in synonyms])
    else:
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(type, device=device)
    if avg_synonyms:
        import itertools
        sentences = list(itertools.chain.from_iterable(sentences_synonyms))

    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        def text_split(text_list, n):
            for i in range(0, len(text_list), n):
                yield text_list[i: i + n]

        text_features = torch.cat([
            model.encode_text(x) for x in text_split(text, 5000)],
            dim=0)

    if prompt == "all":
        num_templates = len(template_list)
        num_channels = text_features.shape[-1]
        text_features = text_features.reshape(num_templates, -1, num_channels).mean(dim=0)

    if avg_synonyms:
        synonyms_per_cat = [len(x) for x in sentences_synonyms]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)

    text_features = text_features.cpu()
    if save_path is not None:
        np.save(open(save_path, 'wb'), text_features.numpy())
    return text_features


@LINEAR_LAYERS.register_module()
class CLIPLinear(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 clip_channels=None,
                 use_sigmoid=True,
                 scale_init=0.0,
                 register=True,
                 clip_cfg=dict(type='RN50', load_from=None),
                 init_cfg=None):
        super(CLIPLinear, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.clip_channels = clip_channels
        self.use_sigmoid = use_sigmoid
        assert use_sigmoid

        # define clip head
        if register:
            load_from = clip_cfg.pop('load_from', None)
            if load_from is not None:
                clip_feat = torch.tensor(np.load(load_from))
            else:
                clip_feat = load_clip_features(**clip_cfg)
            clip_feat = clip_feat.type(torch.float32)
            assert clip_feat.size(0) == out_channels
            clip_feat = F.normalize(clip_feat, p=2, dim=1)
            self.register_buffer("clip_feat", clip_feat)
            self.clip_channels = clip_feat.size(1)
        else:
            self.register_buffer("clip_feat", None)
            self.clip_channels = clip_channels
            assert clip_channels is not None

        # define projection head
        self.image_projection = nn.Identity()
        self.text_projection = nn.Linear(self.clip_channels, in_channels)

        # define and initialize bias
        self.text_bias = nn.Linear(self.clip_channels, 1)

        # define scale
        self.scale = nn.Parameter(torch.Tensor([scale_init]))

        self.init_layers()

    def init_layers(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.text_bias.bias, bias_init)

    def forward(self, x):
        if isinstance(x, tuple):
            x, clip_feat = x
        else:
            clip_feat = self.clip_feat

        text_pred = self.text_projection(clip_feat)
        x_pred = self.image_projection(x)
        bias = self.text_bias(clip_feat)
        x_pred = (x_pred @ text_pred.T) / self.scale.exp() + bias.view(1, -1)
        return x_pred
