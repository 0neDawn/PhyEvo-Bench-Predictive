"""Dataset factory helpers.

Use lazy imports to avoid forcing optional dataset dependencies at startup.
"""


def _load_langtable_datasets():
    try:
        from datasets.langtable_ds import LanguageTableDataset, LanguageTableDatasetImage
    except ImportError as exc:
        raise ImportError(
            "langtable dataset is requested but datasets/langtable_ds.py is missing "
            "or its dependencies are not installed"
        ) from exc
    return LanguageTableDataset, LanguageTableDatasetImage


def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "traffic":
        from datasets.traffic_ds import TrafficDataset
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        from datasets.clevrer_ds import CLEVREREpDataset
        dataset = CLEVREREpDataset(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        from datasets.balls_ds import Balls
        dataset = Balls(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        from datasets.obj3d_ds import Obj3D
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        from datasets.obj3d_ds import Obj3D
        image_size = 128
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        from datasets.phyre_ds import PhyreDataset
        dataset = PhyreDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'langtable':
        LanguageTableDataset, _ = _load_langtable_datasets()
        dataset = LanguageTableDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset


def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # set seq_len > 1 when training with use_tracking
    # load data
    if ds == "traffic":
        from datasets.traffic_ds import TrafficDatasetImage
        dataset = TrafficDatasetImage(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'clevrer':
        from datasets.clevrer_ds import CLEVREREpDatasetImage
        dataset = CLEVREREpDatasetImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        from datasets.balls_ds import BallsImage
        dataset = BallsImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        from datasets.obj3d_ds import Obj3DImage
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        from datasets.obj3d_ds import Obj3DImage
        image_size = 128
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        from datasets.phyre_ds import PhyreDatasetImage
        dataset = PhyreDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'shapes':
        from datasets.shapes_ds import generate_shape_dataset_torch
        if mode == 'train':
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=40_000)
        else:
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=2_000)
    elif ds == 'langtable':
        _, LanguageTableDatasetImage = _load_langtable_datasets()
        dataset = LanguageTableDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset
