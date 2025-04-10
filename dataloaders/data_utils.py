from dataloaders.datasets.esposalles import EsposalledDataset
from dataloaders.datasets.funsd import FUNSDDataset
from dataloaders.datasets.washington import GWDataset
from dataloaders.datasets.historical_maps import HistoricalMapsdDataset
from dataloaders.datasets.iam import IAMDataset
from dataloaders.datasets.iit5k import IIIT5kDataset
from dataloaders.datasets.mlt19 import MLT19Dataset
from dataloaders.datasets.parzival import ParzivalDataset
from dataloaders.datasets.totaltext import TotalTextDataset
from dataloaders.datasets.svt import SVTDataset
from dataloaders.datasets.sroie import SROIEDataset
from dataloaders.datasets.word_art import  WordArtDataset
from dataloaders.datasets.amr import AMRDataset


def load_datasets(args, transforms=lambda x: x, split_langs=True):
    datasets = []

    common = {
        'image_height': args.image_height,
        'patch_width': args.patch_width,
        'transforms': transforms
    }

    bool_lut = {
        'true': True,
        'false': False
    }
    if args.use_word_art:
        datasets.append(
            {
                'train': WordArtDataset(base_location=args.word_art_path, split='train', **common),
                'val': WordArtDataset(base_location=args.word_art_path, split='validation', **common),
                'test': None
             }
        )
    if args.use_amr:
        datasets.append(
            {
                'train': AMRDataset(base_folder=args.amr_path, split='training', **common),
                'val': AMRDataset(base_folder=args.amr_path, split='validation', **common),
                'test': AMRDataset(base_folder=args.amr_path, split='testing', **common),
             }
        )

    if args.use_cocotext:

        if not split_langs:
            raise NotImplementedError

        else:
            raise NotImplementedError

    if args.use_esposalles:
        datasets.append(
            {
                'train': EsposalledDataset(base_folder=args.esposalles_path, split='train',
                                           cross_val=args.esposalles_cross_validation_fold, mode=args.esposalles_level,
                                           **common),
                'test': EsposalledDataset(base_folder=args.esposalles_path, split='test',
                                          cross_val=args.esposalles_cross_validation_fold, mode=args.esposalles_level,
                                          **common),
                'val': None
            }
        )
    if args.use_funsd:
        datasets.append(
            {
                'train': FUNSDDataset(base_folder=args.funsd_path, split='train', **common),
                'test': FUNSDDataset(base_folder=args.funsd_path, split='test', **common),
                'val': None
            }
        )
    if args.use_hiertext:
        raise NotImplementedError

    if args.use_hist_maps:
        datasets.append(
            {
                'train': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split='train',
                                                cross_val=args.hist_maps_cross_validation_fold, **common),
                'test': HistoricalMapsdDataset(base_folder=args.hist_maps_path, split='test',
                                               cross_val=args.hist_maps_cross_validation_fold, **common),
                'val': None
            }
        )

    if args.use_iam:
        datasets.append(
            {
                'train': IAMDataset(base_folder=args.iam_path, split='train', mode=args.iam_level, **common),
                'test': IAMDataset(base_folder=args.iam_path, split='test', mode=args.iam_level, **common),
                'val': IAMDataset(base_folder=args.iam_path, split='val', mode=args.iam_level, **common)
            }
        )

    if args.use_iiit:
        datasets.append(
            {
                'train': IIIT5kDataset(base_folder=args.iiit_path, split='train', **common),
                'test': IIIT5kDataset(base_folder=args.iiit_path, split='test', **common),
                'val': None
            }
        )

    if args.use_mlt:

        if not split_langs:
            datasets.append(
                {
                    'train': MLT19Dataset(base_folder=args.mlt_path, split='train', language=args.mlt19_langs,
                                          cross_val=args.mlt19_cross_validation_fold, **common),
                    'val': MLT19Dataset(base_folder=args.mlt_path, split='val', language=args.mlt19_langs,
                                        cross_val=args.mlt19_cross_validation_fold, **common),
                    'test': None
                }
            )
        else:
            for lang in args.mlt19_langs:
                datasets.append(
                    {
                        'train': MLT19Dataset(base_folder=args.mlt_path, split='train', language=[lang],
                                              cross_val=args.mlt19_cross_validation_fold, **common),
                        'val': MLT19Dataset(base_folder=args.mlt_path, split='val', language=[lang],
                                            cross_val=args.mlt19_cross_validation_fold, **common),
                        'test': None
                    }
                )
    if args.use_parzival:
        datasets.append(
            {
                'train': ParzivalDataset(base_folder=args.parzival_path, split='train', mode=args.parzival_level,
                                         **common),
                'val': ParzivalDataset(base_folder=args.parzival_path, split='valid', mode=args.parzival_level,
                                       **common),
                'test': ParzivalDataset(base_folder=args.parzival_path, split='test', mode=args.parzival_level,
                                        **common)

            }
        )

    if args.use_saint_gall:
        raise NotImplementedError


    if args.use_sroie:
        datasets.append(
            {
                'train': SROIEDataset(base_folder=args.sroie_path, split='train', **common),
                'test': SROIEDataset(base_folder=args.sroie_path, split='test', **common),
                'val': None
            }
        )

    if args.use_svt:
        datasets.append(
            {
                'train': SVTDataset(base_folder=args.svt_path, split='train', **common),
                'test': SVTDataset(base_folder=args.svt_path, split='test', **common),
                'val': None

            }
        )

    if args.use_textocr:
        raise NotImplementedError


    if args.use_totaltext:
        datasets.append(
            {
                'train': TotalTextDataset(base_folder=args.totaltext_path, split='Train', **common),
                'test': TotalTextDataset(base_folder=args.totaltext_path, split='Test', **common),
                'val': None
            }
        )

    if args.use_washington:
        datasets.append(
            {
                'train': GWDataset(base_folder=args.washington_path, split='train',
                                   cross_val=args.washington_cross_validation_fold, mode=args.washington_level,
                                   **common),
                'test': GWDataset(base_folder=args.washington_path, split='test',
                                  cross_val=args.washington_cross_validation_fold, mode=args.washington_level,
                                  **common),
                'val': GWDataset(base_folder=args.washington_path, split='valid',
                                 cross_val=args.washington_cross_validation_fold, mode=args.washington_level, **common)
            }
        )

    if args.use_xfund:

        if not split_langs:
            raise NotImplementedError

        else:
            raise NotImplementedError

    if args.use_copiale:
        raise NotImplementedError

    if args.use_borg:
        raise NotImplementedError

    if args.use_vatican:
        raise NotImplementedError

    return datasets

def log_usage(args, split_langs=False):
    datasets = []

    if args.use_cocotext:

        if not split_langs:
            datasets.append('cocotext')
        else:
            for lang in args.cocotext_langs:
                datasets.append(f"cocotext_{lang}")

    if args.use_esposalles:
        datasets.append('esposalles')

    if args.use_funsd:
        datasets.append('funsd')

    if args.use_hiertext:
        datasets.append('hiertext')

    if args.use_hist_maps:
        datasets.append('hist_maps')

    if args.use_iam:
        datasets.append('iam')

    if args.use_iiit:
        datasets.append('iiit')

    if args.use_mlt:

        if not split_langs:
            datasets.append('mlt')
        else:
            for lang in args.mlt19_langs:
                datasets.append(f"mlt_{lang}")
    if args.use_parzival:
        datasets.append('parzival')

    if args.use_saint_gall:
        datasets.append('saintgall')

    if args.use_sroie:
        datasets.append('sroie')

    if args.use_svt:
        datasets.append('svt')

    if args.use_textocr:
        datasets.append('textocr')

    if args.use_totaltext:
        datasets.append('totaltext')

    if args.use_washington:
        datasets.append('gw')

    if args.use_xfund:

        if not split_langs:
            datasets.append('xfund')
        else:
            for lang in args.xfund_langs:
                datasets.append(f"xfund_{lang}")

    return datasets