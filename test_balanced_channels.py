from src.dataset import build_dataloaders
loaders = build_dataloaders(
    raw_dir='data/raw/lgg-mri-segmentation',
    image_size=256, batch_size=2, num_workers=0, seed=42, balanced=True,
)
train_loader = loaders['train']
for mask_batch, mri_batch in train_loader:
    print(f'Ch0 mean={mri_batch[:,0].mean():.3f}, std={mri_batch[:,0].std():.3f}')
    print(f'Ch1 mean={mri_batch[:,1].mean():.3f}, std={mri_batch[:,1].std():.3f}')
    print(f'Ch2 mean={mri_batch[:,2].mean():.3f}, std={mri_batch[:,2].std():.3f}')
    break
