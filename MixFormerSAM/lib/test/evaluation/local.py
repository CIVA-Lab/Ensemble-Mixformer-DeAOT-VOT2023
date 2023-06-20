from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/got10k_lmdb'
    settings.got10k_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/lasot_lmdb'
    settings.lasot_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/lasot'
    settings.network_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/nfs'
    settings.otb_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/OTB2015'
    settings.prj_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM'
    settings.result_plot_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/test/result_plots'
    settings.results_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/pixstor/data/grzc7/MixFormerSAM'
    settings.segmentation_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/test/segmentation_results'
    settings.tc128_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/trackingNet'
    settings.uav_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/UAV123'
    settings.vot_path = '/mnt/pixstor/data/grzc7/MixFormerSAM/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

