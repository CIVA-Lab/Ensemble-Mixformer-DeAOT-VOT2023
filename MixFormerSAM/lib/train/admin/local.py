class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/pixstor/data/grzc7/MixFormerSAM/pretrained_networks'
        self.lasot_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/lasot'
        self.got10k_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/got10k/train'
        self.lasot_lmdb_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/got10k_lmdb'
        self.trackingnet_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/coco'
        self.coco_lmdb_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/vid'
        self.imagenet_lmdb_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
