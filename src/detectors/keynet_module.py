
import kornia as K

from core import device as global_device
from core import laf2homo, set_args


class keynet_module:
    """
    A deep-learning feature detection module using the KeyNet architecture.

    KeyNet is designed to combine the strengths of traditional geometric 
    detectors (like Hessian or Harris) with the power of CNNs. It uses 
    learned filters to find stable, repeatable points that are 
    optimized for matching across different viewpoints.

    Attributes:
        num_features (int): The number of top-scoring keypoints to extract 
            (default: 8000).
        pretrained (bool): Uses weights trained on the HPatches dataset 
            for state-of-the-art repeatability.
    """
    def __init__(self, device=None, **args):
        self.device = device if device is not None else global_device
        self.single_image = True        
        self.pipeliner = False   
        self.pass_through = False
        self.add_to_cache = True
                
        self.args = {
            'id_more': '',
            'params': {
                'pretrained': True,
                'num_features': 8000,
                },
        }
        
        if 'add_to_cache' in args.keys(): self.add_to_cache = args['add_to_cache']
        
        self.id_string, self.args = set_args('keynet', args, self.args)
        self.detector = K.feature.KeyNetDetector(**self.args['params']).to(device)


    def get_id(self):
        return self.id_string
        

    def finalize(self):
        return

    
    def run(self, **args):
        img = K.io.load_image(args['img'][args['idx']], K.io.ImageLoadType.GRAY32, device=self.device).unsqueeze(0)
        kp, kr = self.detector(img)
        kp, kH = laf2homo(kp.detach().to(self.device).squeeze(0))

        return {'kp': kp, 'kH': kH, 'kr': kr.detach().to(self.device).squeeze(0)}
