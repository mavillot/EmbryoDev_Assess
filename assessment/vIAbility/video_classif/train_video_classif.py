import sys
from pathlib import Path
from torch.optim import AdamW
from video_transformers import VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.config import SEED, PARAM_GRID, PATH_DATASET, PATH_MODELS

backbone = TransformersBackbone("facebook/timesformer-base-finetuned-k400", num_unfrozen_stages=1)

download_ucf6("./")
datamodule = VideoDataModule(
    train_root=PATH_DATASET + f"/Oslo_viability/videos/train",
    val_root=PATH_DATASET + f"/Oslo_viability/videos/test",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=backbone.num_features, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head)

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(datamodule, model, optimizer=optimizer, max_epochs=8)

trainer.fit()
