import os
import random
import itertools as it
import numpy as np
import skimage as ski
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

from .subcellular import make_cell_image, to_shape

def resize_cell_image(image, target_shape):
    """
    Resize a 3-channel cell image with appropriate interpolation for each channel.

    Uses bilinear interpolation for probability channels and nearest neighbor 
    for binary mask channels to preserve the discrete nature of segmentation masks.

    :param image: Input image of shape (H, W, 3) where channel 0 is probability/intensity,
        channels 1 and 2 are binary masks.
    :type image: numpy.ndarray
    :param target_shape: Target shape as (height, width) for the resized image.
    :type target_shape: tuple of int
    :returns: Resized image of shape (target_shape[0], target_shape[1], 3) with
        appropriate interpolation applied to each channel.
    :rtype: numpy.ndarray
    """
    out = np.zeros((target_shape[0], target_shape[1], image.shape[2]), dtype=image.dtype)
    # Probability channel (bilinear)
    out[:,:,0] = ski.transform.resize(image[:,:,0], target_shape, order=1, preserve_range=True, anti_aliasing=True)
    # Binary channels (nearest neighbor)
    out[:,:,1] = ski.transform.resize(image[:,:,1], target_shape, order=0, preserve_range=True, anti_aliasing=False)
    out[:,:,2] = ski.transform.resize(image[:,:,2], target_shape, order=0, preserve_range=True, anti_aliasing=False)
    return out


def major_axis_pca_with_center(mask, center):
    """
    Compute the major axis of a binary mask using PCA with a specified center point.

    Performs principal component analysis on the mask pixels relative to a given 
    center point to determine the major axis of the shape. The axis orientation 
    is normalized to point toward the side with more mass distribution.

    :param mask: 2D binary mask where non-zero values indicate the region of interest.
    :type mask: numpy.ndarray
    :param center: Center point as (y, x) coordinates to use for PCA computation 
        (e.g., nucleus centroid).
    :type center: tuple of float
    :returns: Tuple containing the major axis unit vector and the angle in radians.
    :rtype: tuple (major_axis_vector: numpy.ndarray, angle: float)
    """
    yx = np.argwhere(mask > 0)
    yx_centered = yx - np.array(center)  # center at nucleus
    if len(yx_centered) < 2:
        return np.array([1, 0]), 0.0  # default axis
    cov = np.cov(yx_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    # Ensure consistent orientation: major axis points toward side with more mass
    projections = yx_centered @ major_axis
    if projections.sum() < 0:
        major_axis = -major_axis
    angle = np.arctan2(major_axis[1], major_axis[0])  # angle w.r.t. x-axis
    return major_axis, angle


def align_image(image, center='cell', cell_mask_channel=1, nucleus_channel=2):
    """
    Center and align a 3-channel cell image based on morphological features.

    Centers the image based on the centroid of the largest labeled object and 
    rotates it to align the major axis horizontally. The image is padded as 
    needed and trimmed to remove empty borders.

    :param image: Input image of shape (H, W, 3) containing cell imaging data.
    :type image: numpy.ndarray
    :param center: Centering method: 'cell' to center on cell mask, 'nucleus' to center 
        on nucleus mask. Default is 'cell'.
    :type center: str
    :param cell_mask_channel: Index of the cell mask channel. Default is 1.
    :type cell_mask_channel: int
    :param nucleus_channel: Index of the nucleus mask channel. Default is 2.
    :type nucleus_channel: int
    :returns: Centered and rotated image, possibly larger than input due to padding 
        and rotation operations.
    :rtype: numpy.ndarray
    """
    # Center based on largest labeled cell/nucleus object
    if center == 'cell':
        mask = image[..., cell_mask_channel]
    elif center == 'nucleus':
        mask = image[..., nucleus_channel]
    else:
        raise ValueError("center must be 'cell' or 'nucleus'")
    labeled_objects = ski.measure.label(mask > 0)
    object_regions = ski.measure.regionprops(labeled_objects)
    if not object_regions:
        print(f'No {center}s found, returning original image.')
        return image.copy()
    largest_object = max(object_regions, key=lambda x: x.area)
    cy, cx = largest_object.centroid  # (y, x)
    h, w = image.shape[:2]
    pad_top = int(max(0, (h - cy) - cy))
    pad_bottom = int(max(0, cy - (h - cy)))
    pad_left = int(max(0, (w - cx) - cx))
    pad_right = int(max(0, cx - (w - cx)))
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='constant')
    # pad image to make it square
    if padded.shape[0] != padded.shape[1]:
        size = max(padded.shape[:2])
        pad_h = (size - padded.shape[0]) // 2
        pad_w = (size - padded.shape[1]) // 2
        padded = np.pad(padded, ((pad_h, size-padded.shape[0]-pad_h), (pad_w, size-padded.shape[1]-pad_w), (0,0)), mode='constant')
    # add extra padding to ensure mask is not cropped after rotation
    pad_extra = int(padded.shape[0] / 2 * (np.sqrt(2) - 1))
    padded = np.pad(padded, ((pad_extra, pad_extra), (pad_extra, pad_extra), (0,0)), mode='constant')
    # New centroid after padding
    new_cy = padded.shape[0]//2
    new_cx = padded.shape[1]//2
    # Rotation based on largest labeled cell object
    cell_mask = padded[..., cell_mask_channel]
    # Use centroid of largest nucleus for centering, but largest cell for rotation
    major_axis, angle = major_axis_pca_with_center(cell_mask, (new_cy, new_cx))
    angle_deg = -np.degrees(angle)
    rotated_channels = []
    for c in range(padded.shape[2]):
        order = 0 if np.array_equal(np.unique(padded[...,c]), [0,1]) else 1
        rotated = ski.transform.rotate(padded[...,c], angle=angle_deg, center=(new_cy, new_cx), order=order, preserve_range=True)
        rotated_channels.append(rotated)
    result = np.stack(rotated_channels, axis=-1)
    # trim empty borders
    mask = result[..., cell_mask_channel]
    if np.any(mask > 0):
        min_y, min_x = np.array(np.where(mask > 0)).min(axis=1)
        max_y, max_x = np.array(np.where(mask > 0)).max(axis=1)
        trim_len = np.min([min_y, min_x, result.shape[0]-max_y, result.shape[1]-max_x])
        if trim_len > 0:
            result = result[trim_len:result.shape[0]-trim_len, trim_len:result.shape[1]-trim_len, :]
    return result


def make_NN_training_data(save_path, cell_objects, reference_cell_object, mapped_channel_distributions, channel, center='cell', rescale=True, shape=(64, 64)):
    """
    Generate training data from GW_OT_Cell objects for neural network training.

    Creates paired cell images and their corresponding mapped versions for training 
    deep learning models. Images are aligned, normalized, and saved as numpy arrays.

    :param save_path: Directory path to save processed images. Cell images saved to 
        '<save_path>/cell_images' and mapped images to '<save_path>/mapped_cell_images'.
    :type save_path: str
    :param cell_objects: List of GW_OT_Cell objects or paths to pickled GW_OT_Cell objects.
    :type cell_objects: list
    :param reference_cell_object: Reference cell object or path to pickled reference cell object used 
        as template for mapped distributions.
    :type reference_cell_object: GW_OT_Cell or str
    :param mapped_channel_distributions: Array of mapped protein distributions for each cell.
    :type mapped_channel_distributions: numpy.ndarray
    :param channel: Channel name to use for image processing.
    :type channel: str
    :param center: Centering method for image alignment: 'cell' or 'nucleus'. Default is 'cell'.
    :type center: str
    :param rescale: Whether to rescale images to a fixed size. Default is True.
    :type rescale: bool
    :param shape: Target shape (height, width) for resizing images. Default is (64, 64).
    :type shape: tuple of int
    :returns: None. Images are saved to disk as .npy files.
    :rtype: None
    """
    if not rescale:
        max_size = 0
        for cell_object in cell_objects:
            # Load GW_OT_Cell object if path specified
            if isinstance(cell_object, str):
                pickle.load(open(cell_object, 'rb'))
            cell_image = make_cell_image(cell_object, ['nucleus', channel])
            cell_image = to_shape(cell_image, (max(cell_image.shape[:2]), max(cell_image.shape[:2]), 3))
            cell_image = cell_image[:,:,[1,0,2]]
            cell_image = align_image(cell_image, center=center)
            max_size = max(max_size, max(cell_image.shape[:2]))

    for i, cell_object in enumerate(cell_objects):
        # Load GW_OT_Cell object if path specified
        if isinstance(cell_object, str):
            pickle.load(open(cell_object, 'rb'))
        # make image array from cell object
        cell_image = make_cell_image(cell_object, ['nucleus', channel])
        mapped_cell_object = reference_cell_object.copy()
        mapped_cell_object.intensities[channel] = mapped_channel_distributions[i]
        mapped_cell_image = make_cell_image(mapped_cell_object, ['nucleus', channel])
        # pad image to square
        cell_image = to_shape(cell_image, (max(cell_image.shape[:2]), max(cell_image.shape[:2]), 3))
        mapped_cell_image = to_shape(mapped_cell_image, (max(mapped_cell_image.shape[:2]), max(mapped_cell_image.shape[:2]), 3))
        # reorder channels: channel, binary cell mask, binary nucleus mask
        cell_image = cell_image[:,:,[1,0,2]]
        mapped_cell_image = mapped_cell_image[:,:,[1,0,2]]
        # align image
        cell_image = align_image(cell_image, center=center)
        mapped_cell_image = align_image(mapped_cell_image, center=center)
        # resize image
        if not rescale:
            cell_image = to_shape(cell_image, (max_size, max_size, 3))
            mapped_cell_image = to_shape(mapped_cell_image, (max_size, max_size, 3))
        cell_image = resize_cell_image(cell_image, shape)
        mapped_cell_image = resize_cell_image(mapped_cell_image, shape)
        # make cell_images & mapped_cell_images directories if they don't exist
        if not os.path.exists(os.path.join(save_path, 'cell_images')):
            os.makedirs(os.path.join(save_path, 'cell_images'))
        if not os.path.exists(os.path.join(save_path, 'mapped_cell_images')):
            os.makedirs(os.path.join(save_path, 'mapped_cell_images'))
        # save image
        np.save(os.path.join(save_path, 'cell_images', f'cell_{i}.npy'), cell_image)
        np.save(os.path.join(save_path, 'mapped_cell_images', f'mapped_cell_{i}.npy'), mapped_cell_image)


class EfficientNetFeatureExtractor(nn.Module):
    """
    Feature extractor using EfficientNet backbone for cell image embeddings.

    Adapts a pretrained EfficientNet model to extract fixed-size feature 
    embeddings from cell images. Handles variable input channel numbers 
    and resizes inputs to match EfficientNet requirements.

    :param embedding_size: Size of the output embedding vector. Default is 50.
    :type embedding_size: int, optional
    :param input_channels: Number of input channels in the cell images. Default is 3.
    :type input_channels: int, optional
    :param efficientnet_type: Type of EfficientNet architecture to use. Default is 'efficientnet_b0'.
    :type efficientnet_type: str, optional
    :param pretrained: Whether to use pretrained ImageNet weights. Default is True.
    :type pretrained: bool, optional
    """
    def __init__(self, embedding_size=50, input_channels=3, efficientnet_type='efficientnet_b0', pretrained=True):
        super().__init__()
        # Load a pretrained EfficientNet
        efficientnet = getattr(models, efficientnet_type)(pretrained=pretrained)
        # Determine expected input size from model metadata if available
        if hasattr(efficientnet, 'default_cfg') and 'input_size' in efficientnet.default_cfg:
            self.efficientnet_input_size = efficientnet.default_cfg['input_size'][-1]
        else:
            self.efficientnet_input_size = 224  # Fallback for older torchvision
        if input_channels != 3:
            efficientnet.features[0][0] = nn.Conv2d(input_channels, efficientnet.features[0][0].out_channels,
                                                    kernel_size=efficientnet.features[0][0].kernel_size,
                                                    stride=efficientnet.features[0][0].stride,
                                                    padding=efficientnet.features[0][0].padding,
                                                    bias=False)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        self.fc = nn.Linear(efficientnet.classifier[1].in_features, embedding_size)

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Feature embedding tensor of shape (batch_size, embedding_size).
        """
        # Always resize input to expected EfficientNet input size
        if x.shape[2] != self.efficientnet_input_size or x.shape[3] != self.efficientnet_input_size:
            x = F.interpolate(x, size=(self.efficientnet_input_size, self.efficientnet_input_size), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class UNetDecoder(nn.Module):
    """
    U-Net style decoder for reconstructing images from feature embeddings.

    Implements a decoder network that reconstructs images from compressed 
    feature embeddings using transposed convolutions and skip connections.
    The architecture progressively upsamples from a compact representation 
    back to full image resolution.

    :param embedding_size: Size of the input embedding vector. Default is 50.
    :type embedding_size: int, optional
    :param image_size: Target output image size (assumed square). Default is 64.
    :type image_size: int, optional
    :param out_channels: Number of output channels in the reconstructed image. Default is 1.
    :type out_channels: int, optional
    """
    def __init__(self, embedding_size=50, image_size=64, out_channels=1):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Linear(embedding_size, 128 * (image_size // 8) * (image_size // 8))
        # Encoder/decoder blocks
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        """
        Forward pass through the U-Net decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tensor of shape (batch_size, embedding_size).

        Returns
        -------
        torch.Tensor
            Reconstructed image tensor of shape (batch_size, out_channels, 
            image_size, image_size) with softmax-normalized probability 
            distributions.
        """
        # x: (batch, embedding_size)
        x = self.fc(x)
        x = x.view(x.size(0), 128, self.image_size // 8, self.image_size // 8)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.final(x)
        # Output: (batch, out_channels, H, W)
        x = torch.softmax(x.view(x.size(0), -1), dim=1).view(x.size(0), 1, self.image_size, self.image_size)
        return x


class dGWOTNetwork(nn.Module):
    """
    Deep Gromov-Wasserstein Optimal Transport Network.
    
    A complete neural network architecture that combines feature extraction, 
    distance computation, and image reconstruction in a multi-task learning 
    framework. Designed for learning embeddings that preserve Gromov-Wasserstein 
    distances between cell morphologies while enabling reconstruction of 
    protein distributions.

    :param input_channels: Number of input image channels. Default is 3.
    :type input_channels: int, optional
    :param embedding_size: Dimensionality of the feature embedding space. Default is 50.
    :type embedding_size: int, optional
    :param image_size: Size of input/output images (assumed square). Default is 64.
    :type image_size: int, optional
    """
    def __init__(self, input_channels=3, embedding_size=50, image_size=64):
        super().__init__()
        self.feature_extractor = EfficientNetFeatureExtractor(
            embedding_size=embedding_size,
            input_channels=input_channels,
            efficientnet_type='efficientnet_b4',  
            pretrained=True
        )
        # Only decode the first/protein/probability channel
        self.feature_decoder = UNetDecoder(embedding_size, image_size)
        
    def forward(self, x1, x2, return_embedding=False):
        """
        Forward pass through the complete dGWOT network.

        Processes two input images through feature extraction, computes their 
        embedding distance, and reconstructs both images. Optionally returns 
        the intermediate feature embeddings.

        Parameters
        ----------
        x1, x2 : torch.Tensor
            Input image tensors of shape (batch_size, channels, height, width).
        return_embedding : bool, optional
            Whether to return intermediate feature embeddings. Default is False.

        Returns
        -------
        tuple
            If return_embedding is False:
                distance : torch.Tensor
                    Squared Euclidean distance between embeddings of shape 
                    (batch_size, 1).
                uf1, uf2 : torch.Tensor
                    Reconstructed images of shape (batch_size, 1, height, width).
            
            If return_embedding is True:
                distance : torch.Tensor
                    Squared Euclidean distance between embeddings.
                uf1, uf2 : torch.Tensor
                    Reconstructed images.
                feat1, feat2 : torch.Tensor
                    Feature embeddings of shape (batch_size, embedding_size).
        """
        # Extract features from both images
        feat1 = self.feature_extractor(x1)
        feat2 = self.feature_extractor(x2)
        # Compute Euclidean distance in embedding space
        distance = torch.sum((feat1 - feat2) ** 2, dim=1, keepdim=True)
        # Reconstruct both images from their embeddings
        uf1 = self.feature_decoder(feat1)
        uf2 = self.feature_decoder(feat2)
        # Return: distance, reconstruction1, reconstruction2, copy1, copy2
        if return_embedding:
            return distance, uf1, uf2, feat1, feat2
        else:
            return distance, uf1, uf2
        

class PretrainPairedDataset(Dataset):
    """
    PyTorch Dataset for pretraining with paired input and target images.

    Loads pairs of numpy arrays for pretraining tasks where each input image 
    has a corresponding target image. Handles channel dimension reordering 
    and applies optional transforms.

    :param input_files: List of file paths to input image numpy arrays.
    :type input_files: list of str
    :param target_files: List of file paths to target image numpy arrays. Must have same 
        length as input_files.
    :type target_files: list of str
    :param transform: Optional transform to apply to both input and target images.
        Default is None.
    :type transform: callable, optional
    :param augment_transform: Additional augmentation transform for data augmentation. Default is None.
    :type augment_transform: callable, optional

    :raises AssertionError: If input_files and target_files have different lengths.
    """
    def __init__(self, input_files, target_files, transform=None, augment_transform=None, n_augment=1):
        self.input_files = [f for f in input_files for _ in range(n_augment)]
        self.target_files = [f for f in target_files for _ in range(n_augment)]
        assert len(self.input_files) == len(self.target_files), 'Input and target directories must have the same number of images.'
        self.transform = transform
        self.augment_transform = augment_transform
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_img = np.load(self.input_files[idx])
        target_img = np.load(self.target_files[idx])
        # Ensure shape is (C, H, W) for torch
        if input_img.ndim == 2:
            input_img = input_img[np.newaxis, ...]
        elif input_img.ndim == 3 and input_img.shape[0] != 3 and input_img.shape[-1] == 3:
            input_img = np.transpose(input_img, (2, 0, 1))
        if target_img.ndim == 2:
            target_img = target_img[np.newaxis, ...]
        elif target_img.ndim == 3 and target_img.shape[0] != 3 and target_img.shape[-1] == 3:
            target_img = np.transpose(target_img, (2, 0, 1))
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        if self.augment_transform:
            input_img = self.augment_transform(input_img)
            target_img = self.augment_transform(target_img)
        input_img = torch.from_numpy(input_img).float()
        target_img = torch.from_numpy(target_img).float()
        return input_img, target_img


def pretrain_model(paired_dataset, model, save_path=None, model_name="pretrained_model", 
                  batch_size=64, epochs=10, lr=1e-3, device=None, return_model=True):
    """
    Pretrain a model using paired input and target images provided as a PairedDataset.

    This function accepts a `PairedDataset` object and first converts it into a
    `PretrainPairedDataset` by collecting the unique image indices referenced in
    the paired dataset. For each unique image index `i`, the input path is
    '<image_dir>/cell_i.npy' and the target path is
    '<mapped_image_dir>/mapped_cell_i.npy'. After conversion the rest of the
    training loop is identical to the previous implementation.

    :param paired_dataset: Dataset containing paired indices and directory information. Must have
        attributes `image_dir`, `mapped_image_dir` and `image_pairs`.
    :type paired_dataset: PairedDataset
    :param model: Neural network model to pretrain. Must have a forward method that 
        takes two identical inputs and returns reconstructions.
    :type model: torch.nn.Module
    :param save_path: Directory path to save the pretrained model. If None, model is not saved.
        Default is None.
    :type save_path: str, optional
    :param model_name: Name prefix for saved model files. Default is "pretrained_model".
    :type model_name: str, optional
    :param batch_size: Batch size for training. Default is 64.
    :type batch_size: int, optional
    :param epochs: Number of training epochs. Default is 10.
    :type epochs: int, optional
    :param lr: Learning rate for the Adam optimizer. Default is 1e-3.
    :type lr: float, optional
    :param device: Device to run training on. If None, automatically selects GPU 
        if available. Default is None.
    :type device: torch.device, optional
    :param return_model: Whether to return the trained model. If False, returns None.
        Default is True.
    :type return_model: bool, optional

    :returns: The pretrained model if return_model is True, otherwise None.
    :rtype: torch.nn.Module or None
    """
    # Convert PairedDataset to lists of input/target files (unique images)
    if not isinstance(paired_dataset, PairedDataset):
        raise TypeError("paired_dataset must be an instance of PairedDataset")

    all_indices = set()
    for pair in paired_dataset.image_pairs:
        all_indices.update(pair)
    all_indices = sorted(list(all_indices))

    input_files = [os.path.join(paired_dataset.image_dir, f"cell_{idx}.npy") for idx in all_indices]
    target_files = [os.path.join(paired_dataset.mapped_image_dir, f"mapped_cell_{idx}.npy") for idx in all_indices]

    dataset = PretrainPairedDataset(input_files, target_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Store config for saving (if it's a dGWOTNetwork)
    config = None
    if hasattr(model, 'feature_extractor') and hasattr(model, 'feature_decoder'):
        # Try to extract config from dGWOTNetwork
        try:
            input_channels = model.feature_extractor.feature_extractor.features[0][0].in_channels if hasattr(model.feature_extractor, 'feature_extractor') else 3
            embedding_size = model.feature_extractor.fc.out_features
            image_size = model.feature_decoder.image_size
            config = {
                'input_channels': input_channels,
                'embedding_size': embedding_size,
                'image_size': image_size
            }
        except:
            print("Warning: Could not extract model config for saving")
    
    # Create save directory if specified
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for input_batch, target_batch in progress_bar:
            input_batch = input_batch.float()
            target_batch = target_batch.float()
            if input_batch.ndim == 4 and input_batch.shape[-1] == 3:
                input_batch = input_batch.permute(0, 3, 1, 2)
            if target_batch.ndim == 4 and target_batch.shape[-1] == 3:
                target_batch = target_batch.permute(0, 3, 1, 2)
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            _, recon, _ = model(input_batch, input_batch)
            # Only reconstruct the first channel (probability/protein) of the target
            target = target_batch[:, 0:1, :, :]
            pred = recon[:, 0:1, :, :]
            loss = kullback_leibler_divergence_loss(target, pred)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * input_batch.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # Save best model if save_path is provided
        if save_path is not None and epoch_loss < best_loss:
            best_loss = epoch_loss
            if config is not None:
                # Save with config
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch + 1,
                    'loss': epoch_loss
                }, os.path.join(save_path, f'{model_name}_best.pth'))
            else:
                # Save without config (backward compatibility)
                torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_best.pth'))
            print(f'    → New best model saved (loss: {epoch_loss:.6f})')
    
    # Save final model if save_path is provided
    if save_path is not None:
        if config is not None:
            torch.save({
                'state_dict': model.state_dict(),
                'config': config,
                'epoch': epochs,
                'loss': epoch_loss
            }, os.path.join(save_path, f'{model_name}_final.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_final.pth'))
        print(f'Saved pretrained model to {save_path}/{model_name}_final.pth')
    
    return model if return_model else None


def kullback_leibler_divergence_loss(y_true, y_pred):
    """
    Compute Kullback-Leibler divergence loss for probability distributions.

    Measures how well a predicted probability distribution matches a target 
    distribution. Used for reconstruction quality assessment when dealing 
    with normalized protein distributions.

    :param y_true: Target probability distribution tensor.
    :type y_true: torch.Tensor
    :param y_pred: Predicted probability distribution tensor.
    :type y_pred: torch.Tensor
    :returns: Mean KL divergence loss across the batch.
    :rtype: torch.Tensor

    .. note::

        The KL divergence is computed as: KL(P||Q) = sum(P * log(P/Q))
        Values are clamped to avoid log(0) numerical issues.
    """
    epsilon = 1e-8
    
    # Clamp values to avoid log(0)
    y_true = torch.clamp(y_true, epsilon, 1.0)
    y_pred = torch.clamp(y_pred, epsilon, 1.0)
    
    # Flatten tensors for computation
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_div = torch.sum(y_true_flat * torch.log(y_true_flat / y_pred_flat), dim=1)
    return torch.mean(kl_div)


def sparsity_constraint_loss(embeddings, sparsity_target=0.1):
    """
    Apply KL divergence sparsity constraint to hidden unit activations.

    Encourages sparse representations by penalizing deviations from a target 
    sparsity level. This regularization helps prevent overfitting and promotes 
    more interpretable feature representations.

    :param embeddings: Hidden unit activations of shape (batch_size, embedding_dim).
    :type embeddings: torch.Tensor
    :param sparsity_target: Desired average activation level for each hidden unit. Default is 0.1.
    :type sparsity_target: float, optional
    :returns: Scalar sparsity loss computed as the sum of KL divergences across 
        all embedding dimensions.
    :rtype: torch.Tensor

    .. note::

        Activations are passed through sigmoid to ensure they're in (0,1) range 
        before computing the sparsity constraint.
    """
    epsilon = 1e-8
    # Apply sigmoid to ensure activations are in (0,1)
    activations = torch.sigmoid(embeddings)
    rho_hat = torch.mean(activations, dim=0)  # (embedding_dim,)
    rho = torch.full_like(rho_hat, sparsity_target)
    kl = rho * torch.log((rho + epsilon) / (rho_hat + epsilon)) + \
         (1 - rho) * torch.log((1 - rho + epsilon) / (1 - rho_hat + epsilon))
    return torch.sum(kl)


def reconstruction_loss(x, uf):
    """
    Compute multi-channel reconstruction loss for cell images.

    Applies appropriate loss functions for different channel types:
    probability distributions use KL divergence, while binary masks 
    use binary cross-entropy loss.

    :param x: Original image tensor of shape (batch_size, 3, height, width).
    :type x: torch.Tensor
    :param uf: Reconstructed image tensor of same shape as x.
    :type uf: torch.Tensor
    :returns: Combined reconstruction loss across all channels.
    :rtype: torch.Tensor

    .. note::

        - Channel 0: Probability distribution (KL divergence loss)
        - Channel 1: Binary cell mask (Binary cross-entropy loss)
        - Channel 2: Binary nucleus mask (Binary cross-entropy loss)
    """
    # Split channels
    x_prob, x_mask1, x_mask2 = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
    uf_prob, uf_mask1, uf_mask2 = uf[:, 0:1, :, :], uf[:, 1:2, :, :], uf[:, 2:3, :, :]

    # Probability channel: KL divergence
    kl = kullback_leibler_divergence_loss(x_prob, uf_prob)
    # Binary mask channels: BCE loss
    bce1 = F.binary_cross_entropy(uf_mask1, x_mask1)
    bce2 = F.binary_cross_entropy(uf_mask2, x_mask2)
    return kl + bce1 + bce2


def get_random_pairs(indices, n_pairs):
    """
    Generate a random subset of unique pairs from a list of indices.

    Creates all possible unique pairs from the input indices and randomly 
    samples a specified number of them. Useful for creating training pairs 
    from a dataset without exhaustive pairwise combinations.

    :param indices: List or array of indices to create pairs from.
    :type indices: array-like
    :param n_pairs: Number of pairs to randomly sample. If larger than the total 
        possible pairs, returns all possible pairs.
    :type n_pairs: int
    :returns: Array of shape (n_pairs, 2) containing randomly selected index pairs.
    :rtype: numpy.ndarray
    """
    all_pairs = np.array(list(it.combinations(indices, 2)))
    if n_pairs > len(all_pairs):
        n_pairs = len(all_pairs)
    pair_inds = np.random.choice(len(all_pairs), n_pairs, replace=False)
    return all_pairs[pair_inds]


class IndexedImageDataset(Dataset):
    """
    PyTorch Dataset for loading individual cell images by index.

    Simple dataset for loading cell images by index, useful for extracting 
    embeddings from unique images in a PairedDataset without loading duplicates.

    :param image_dir: Path to directory containing cell image .npy files with naming 
        convention 'cell_{index}.npy'.
    :type image_dir: str
    :param indices: List of cell indices to load.
    :type indices: list of int
    :param transform: Transform function to apply to all images. Default is None.
    :type transform: callable, optional
    """
    def __init__(self, image_dir, indices, transform=None):
        self.image_dir = image_dir
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        img_path = os.path.join(self.image_dir, f"cell_{index}.npy")
        image = np.load(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image


class PairedDataset(Dataset):
    """
    PyTorch Dataset for loading paired cell images with distance labels.

    Loads pairs of cell images and their mapped counterparts from numpy files 
    for training distance-based models. Supports data augmentation and lazy 
    loading for memory efficiency.

    :param image_dir: Path to directory containing cell image .npy files with naming 
        convention 'cell_{index}.npy'.
    :type image_dir: str
    :param mapped_image_dir: Path to directory containing mapped cell image .npy files with naming 
        convention 'mapped_cell_{index}.npy'.
    :type mapped_image_dir: str
    :param distances: Distance values corresponding to each image pair for supervised learning.
    :type distances: list of float
    :param image_pairs: List of (index1, index2) tuples specifying which images to pair.
    :type image_pairs: list of tuple
    :param transform: Transform function to apply to all images. Default is None.
    :type transform: callable, optional
    :param augment_transform: Additional augmentation transform for data augmentation. Default is None.
    :type augment_transform: callable, optional
    :param n_augment: Number of augmented copies to create for each pair. Default is 1.
    :type n_augment: int, optional

    .. note::

        The dataset expects file naming conventions:
        - Cell images: 'cell_{index}.npy'
        - Mapped images: 'mapped_cell_{index}.npy'
    """
    def __init__(self, image_dir, mapped_image_dir, distances, image_pairs, transform=None, augment_transform=None, n_augment=1):
        # Store directory paths. Listing all files is no longer needed.
        self.image_dir = image_dir
        self.mapped_image_dir = mapped_image_dir

        # The rest of the logic remains the same
        self.distances = [distance for distance in distances for _ in range(n_augment)]
        self.image_pairs = [image_pair for image_pair in image_pairs for _ in range(n_augment)]
        self.transform = transform
        self.augment_transform = augment_transform

    def __getitem__(self, index):
        # Get the integer indices for the pair
        ind_1, ind_2 = self.image_pairs[index]
        
        # Dynamically construct the filenames using the indices
        img_path_1 = os.path.join(self.image_dir, f"cell_{ind_1}.npy")
        img_path_2 = os.path.join(self.image_dir, f"cell_{ind_2}.npy")
        mapped_img_path_1 = os.path.join(self.mapped_image_dir, f"mapped_cell_{ind_1}.npy")
        mapped_img_path_2 = os.path.join(self.mapped_image_dir, f"mapped_cell_{ind_2}.npy")

        # Load the NumPy arrays from the .npy files
        image_1 = np.load(img_path_1)
        image_2 = np.load(img_path_2)
        mapped_image_cell_1 = np.load(mapped_img_path_1)
        mapped_image_cell_2 = np.load(mapped_img_path_2)
        
        # Apply the general transform if it exists
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            mapped_image_cell_1 = self.transform(mapped_image_cell_1)
            mapped_image_cell_2 = self.transform(mapped_image_cell_2)

        # Apply the augmentation transform if it exists
        if self.augment_transform is not None:
            image_1 = self.augment_transform(image_1)
            image_2 = self.augment_transform(image_2)
        
        distance = self.distances[index]

        return image_1, image_2, mapped_image_cell_1, mapped_image_cell_2, distance

    def __len__(self):
        # The length is determined by the number of pairs, which is correct
        return len(self.image_pairs)
    

class RandomHorizontalRescale(object):
    """
    Data augmentation transform that randomly rescales image width.

    Randomly rescales the horizontal axis of cell images to achieve uniform 
    distribution of cell mask widths. Uses appropriate interpolation methods 
    for different channel types (bilinear for intensity, nearest for masks).

    :param min_relative_width: Minimum relative width of the cell mask as fraction of image width.
        Default is 0.1.
    :type min_relative_width: float, optional
    :param max_relative_width: Maximum relative width of the cell mask as fraction of image width.
        Default is 1.0.
    :type max_relative_width: float, optional

    .. note::

        - Channel 0: Resized with bilinear interpolation (intensity/probability)
        - Other channels: Resized with nearest neighbor interpolation (binary masks)
        The transform maintains the original image width by padding or cropping after rescaling.
    """
    def __init__(self, min_relative_width=0.1, max_relative_width=1.0):
        assert 0 < min_relative_width <= max_relative_width <= 1.0
        self.min_relative_width = min_relative_width
        self.max_relative_width = max_relative_width

    def __call__(self, image):
        c, h, w = image.shape
        mask = image[1]  # channel 1 is the segmentation mask
        mask_inds = (mask > 0).nonzero(as_tuple=False)
        min_x = mask_inds[:, 1].min().item()
        max_x = mask_inds[:, 1].max().item()
        mask_width = max_x - min_x + 1

        # Compute allowed min/max mask widths in pixels
        min_mask_width = int(self.min_relative_width * w)
        max_mask_width = int(self.max_relative_width * w)
        min_mask_width = max(1, min_mask_width)
        max_mask_width = max(min_mask_width, max_mask_width)

        # Sample target mask width
        target_mask_width = random.randint(min_mask_width, max_mask_width)
        scale = target_mask_width / mask_width

        # Compute new width for the whole image
        new_w = int(round(w * scale))
        new_w = max(1, new_w)

        # Resize each channel separately, with bilinear for channel 0, nearest for others
        resized = []
        for i in range(c):
            channel = image[i].unsqueeze(0)
            if i == 0:
                resized_channel = TF.resize(channel, [h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
            else:
                resized_channel = TF.resize(channel, [h, new_w], interpolation=TF.InterpolationMode.NEAREST)
            resized.append(resized_channel.squeeze(0))
        image_rescaled = torch.stack(resized, dim=0)

        # Pad or crop to original width
        if new_w < w:
            pad = (w - new_w) // 2
            image_rescaled = TF.pad(image_rescaled, (pad, 0, w - new_w - pad, 0))
        elif new_w > w:
            crop = (new_w - w) // 2
            image_rescaled = image_rescaled[:, :, crop:crop + w]
        return image_rescaled
    

def train_dGWOT(train_dataset, valid_dataset, test_dataset, save_path, dataset_name, embedding_size=50, 
              image_shape=(64,64), batch_size=100, epochs=100, 
              device=None, learning_rate=0.001, dist_weight=1.0,
              early_stopping=True, patience=3, weight_decay=1e-5,
              lr_gamma=0.95, sparsity_weight=0.0, sparsity_target=0.05,
              pretrained_path=None, show_loss_components=False):
    """
    Train the Deep Gromov-Wasserstein Optimal Transport model.

    Trains a dGWOT network using multi-task learning with distance prediction 
    and image reconstruction objectives. Supports early stopping, learning rate 
    scheduling, and optional sparsity constraints.

    :param train_dataset: PyTorch datasets for training, validation, and testing.
    :type train_dataset: Dataset
    :param valid_dataset: Validation dataset.
    :type valid_dataset: Dataset
    :param test_dataset: Test dataset.
    :type test_dataset: Dataset
    :param save_path: Directory path to save the trained model and checkpoints.
    :type save_path: str
    :param dataset_name: Name prefix for saved model files.
    :type dataset_name: str
    :param embedding_size: Dimensionality of the feature embedding space. Default is 50.
    :type embedding_size: int, optional
    :param image_shape: Shape of input images as (height, width). Default is (64, 64).
    :type image_shape: tuple of int, optional
    :param batch_size: Batch size for training. Default is 100.
    :type batch_size: int, optional
    :param epochs: Maximum number of training epochs. Default is 100.
    :type epochs: int, optional
    :param device: Device for training. If None, automatically selects GPU if available.
    :type device: torch.device, optional
    :param learning_rate: Initial learning rate for Adam optimizer. Default is 0.001.
    :type learning_rate: float, optional
    :param dist_weight: Weight for distance loss vs reconstruction loss in total loss. Default is 1.0.
    :type dist_weight: float, optional
    :param early_stopping: Whether to use early stopping based on validation loss. Default is True.
    :type early_stopping: bool, optional
    :param patience: Number of epochs to wait for improvement before stopping. Default is 3.
    :type patience: int, optional
    :param weight_decay: L2 regularization weight for optimizer. Default is 1e-5.
    :type weight_decay: float, optional
    :param lr_gamma: Decay factor for exponential learning rate scheduler. Default is 0.95.
    :type lr_gamma: float, optional
    :param sparsity_weight: Weight for sparsity constraint loss. Default is 0.0 (disabled).
    :type sparsity_weight: float, optional
    :param sparsity_target: Target sparsity level for hidden activations. Default is 0.05.
    :type sparsity_target: float, optional
    :param pretrained_path: Path to pretrained model weights to initialize from. Default is None.
    :type pretrained_path: str, optional
    :param show_loss_components: Whether to display individual loss components (distance, reconstruction, 
        sparsity) during training. Default is False.
    :type show_loss_components: bool, optional

    :returns: Tuple containing the trained model, training loss history, and validation loss history.
    :rtype: tuple (model: torch.nn.Module, train_losses: list of float, val_losses: list of float)
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_channels = 3
    image_size = image_shape[0]
    model = dGWOTNetwork(input_channels, embedding_size, image_size).to(device)
    
    # Store config for saving
    config = {
        'input_channels': input_channels,
        'embedding_size': embedding_size,
        'image_size': image_size
    }
    
    # Load pretrained weights if provided
    if pretrained_path is not None and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create model directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Starting training for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_idx, (x1, x2, mapped_x1, mapped_x2, target_dist) in train_iter:
            x1, x2, mapped_x1, mapped_x2, target_dist = x1.to(device), x2.to(device), mapped_x1.to(device), mapped_x2.to(device), target_dist.to(device)
            x1 = torch.clamp(x1, 0.0, 1.0)
            x2 = torch.clamp(x2, 0.0, 1.0)
            mapped_x1 = torch.clamp(mapped_x1, 0.0, 1.0)
            mapped_x2 = torch.clamp(mapped_x2, 0.0, 1.0)
            optimizer.zero_grad()
            # Forward pass (with embeddings)
            distance, uf1, uf2, emb1, emb2 = model(x1, x2, return_embedding=True)
            dist_loss = F.mse_loss(distance.squeeze(), target_dist)
            mapped_x1_prob = mapped_x1[:, 0:1, :, :]
            mapped_x2_prob = mapped_x2[:, 0:1, :, :]
            kl1 = kullback_leibler_divergence_loss(mapped_x1_prob, uf1)
            kl2 = kullback_leibler_divergence_loss(mapped_x2_prob, uf2)
            recon_loss = kl1 + kl2
            # KL sparsity constraint
            sparsity_loss = sparsity_constraint_loss(emb1, sparsity_target) + sparsity_constraint_loss(emb2, sparsity_target)
            total_loss = dist_weight * dist_loss + recon_loss + sparsity_weight * sparsity_loss
            total_loss.backward()
            optimizer.step()
            batch_size_actual = x1.size(0)
            train_loss += total_loss.item() * batch_size_actual
            train_samples += batch_size_actual
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for x1, x2, mapped_x1, mapped_x2, target_dist in valid_loader:
                x1, x2, mapped_x1, mapped_x2, target_dist = x1.to(device), x2.to(device), mapped_x1.to(device), mapped_x2.to(device), target_dist.to(device)
                distance, uf1, uf2 = model(x1, x2)
                dist_loss = F.mse_loss(distance.squeeze(), target_dist)
                mapped_x1_prob = mapped_x1[:, 0:1, :, :]
                mapped_x2_prob = mapped_x2[:, 0:1, :, :]
                kl1 = kullback_leibler_divergence_loss(mapped_x1_prob, uf1)
                kl2 = kullback_leibler_divergence_loss(mapped_x2_prob, uf2)
                recon_loss = kl1 + kl2
                total_loss = dist_weight * dist_loss + recon_loss
                batch_size_actual = x1.size(0)
                val_loss += total_loss.item() * batch_size_actual
                val_samples += batch_size_actual
        # Step the learning rate scheduler
        scheduler.step()
        # Calculate average losses per pair
        train_loss /= train_samples
        val_loss /= val_samples
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, '
              f'Val Loss: {val_loss:.6f}')
        
        if show_loss_components:
            # check ranges of losses (FOR TESTING)
            print(f"Distance Loss: {dist_loss.item()}, Reconstruction Loss: {recon_loss.item()}, Sparsity Loss: {sparsity_loss.item()}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model with config
            torch.save({
                'state_dict': model.state_dict(),
                'config': config
            }, f'{save_path}/{dataset_name}_best.pth')
            print(f'    → New best model saved (val_loss: {val_loss:.6f})')
        else:
            patience_counter += 1
            if early_stopping and patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (patience: {patience})')
                break
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for x1, x2, mapped_x1, mapped_x2, target_dist in test_loader:
            x1, x2, mapped_x1, mapped_x2, target_dist = x1.to(device), x2.to(device), mapped_x1.to(device), mapped_x2.to(device), target_dist.to(device)
            distance, uf1, uf2 = model(x1, x2)
            dist_loss = F.mse_loss(distance, target_dist)
            mapped_x1_prob = mapped_x1[:, 0:1, :, :]
            mapped_x2_prob = mapped_x2[:, 0:1, :, :]
            kl1 = kullback_leibler_divergence_loss(mapped_x1_prob, uf1)
            kl2 = kullback_leibler_divergence_loss(mapped_x2_prob, uf2)
            recon_loss = kl1 + kl2
            total_loss = dist_weight * dist_loss + recon_loss
            batch_size_actual = x1.size(0)
            test_loss += total_loss.item() * batch_size_actual
            test_samples += batch_size_actual
    
    test_loss /= test_samples
    print(f'Final Test Loss: {test_loss:.6f}')
    
    # Save final model with config
    print("\nSaving model...")
    torch.save({
        'state_dict': model.state_dict(),
        'config': config
    }, f'{save_path}/{dataset_name}_final.pth')
    print(f'Saved DWE model to {save_path}/{dataset_name}_final.pth')
    
    return model, train_losses, val_losses


def load_dGWOT_model(checkpoint_path, device=None):
    """
    Load a dGWOT model from a checkpoint containing state dict and config.
    
    :param checkpoint_path: Path to the checkpoint file containing both state_dict and config.
    :type checkpoint_path: str
    :param device: Device to load the model on. If None, uses GPU if available.
    :type device: torch.device, optional
    :returns: Loaded dGWOT model ready for inference or further training.
    :rtype: torch.nn.Module
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint and 'state_dict' in checkpoint:
            # New format with config
            config = checkpoint['config']
            state_dict = checkpoint['state_dict']
            
            # Create model with saved config
            model = dGWOTNetwork(**config)
            model.load_state_dict(state_dict)
            
            print(f"Loaded dGWOT model from {checkpoint_path}")
            print(f"Config: {config}")
            
        elif 'state_dict' in checkpoint:
            # Old format with just state_dict
            print("Warning: Loading checkpoint without config. You'll need to specify model parameters manually.")
            return checkpoint['state_dict']
        else:
            # Assume checkpoint is a bare state_dict
            print("Warning: Loading bare state_dict. You'll need to specify model parameters manually.")
            return checkpoint
    else:
        print("Warning: Unknown checkpoint format.")
        return checkpoint
    
    # Move to device
    model = model.to(device)
    
    return model


def extract_embeddings(model, data, batch_size=64, device=None):
    """
    Extract latent embeddings from a trained dGWOT model.

    Processes input images through the feature extractor to obtain latent 
    embeddings. Supports lists/arrays of images, PyTorch datasets, and PairedDatasets.

    :param model: Trained dGWOT model with a feature_extractor attribute.
    :type model: torch.nn.Module
    :param data: Input data to extract embeddings from. Can be:
        - List of numpy arrays with shape (H, W, C)
        - Single numpy array with shape (N, H, W, C)
        - PyTorch Dataset where __getitem__ returns images
        - PairedDataset (extracts embeddings for unique images only)
    :type data: list, numpy.ndarray, torch.utils.data.Dataset, or PairedDataset
    :param batch_size: Batch size for processing. Default is 64.
    :type batch_size: int, optional
    :param device: Device to run computation on. If None, uses model's current device.
    :type device: torch.device, optional
    :returns: Extracted embeddings of shape (N, embedding_size) where N is the 
        number of input images.
    :rtype: numpy.ndarray
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    embeddings = []
    
    # Case 1: PairedDataset - extract embeddings for unique images only
    if isinstance(data, PairedDataset):
        # Get all unique image indices from the dataset pairs
        all_indices = set()
        for pair in data.image_pairs:
            all_indices.update(pair)
        all_indices = sorted(list(all_indices))
        
        print(f"Extracting embeddings for {len(all_indices)} unique images from PairedDataset...")
        
        # Create dataset for unique images
        unique_image_dataset = IndexedImageDataset(
            data.image_dir, 
            all_indices, 
            transform=data.transform
        )
        
        # Process the unique image dataset
        loader = DataLoader(unique_image_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                images = batch
                images = images.to(device)
                if images.dtype != torch.float32:
                    images = images.float()
                
                # Extract features using the model's feature extractor
                feats = model.feature_extractor(images)
                embeddings.append(feats.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    # Case 2: General PyTorch Dataset
    elif hasattr(data, '__getitem__') and hasattr(data, '__len__') and isinstance(data, Dataset):
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting embeddings"):
                # Handle different dataset return formats
                if isinstance(batch, (list, tuple)):
                    # If dataset returns multiple items, take the first (assumed to be images)
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(device)
                if images.dtype != torch.float32:
                    images = images.float()
                
                # Extract features using the model's feature extractor
                feats = model.feature_extractor(images)
                embeddings.append(feats.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    # Case 2: List or numpy array of images
    # Convert list to numpy array if needed
    if isinstance(data, list):
        data = np.stack(data, axis=0)
    
    # Ensure data is numpy array with shape (N, H, W, C)
    if data.ndim == 3:
        data = data[np.newaxis, ...]  # Add batch dimension
    
    n_images = data.shape[0]
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, n_images, batch_size), desc="Extracting embeddings"):
            batch_end = min(i + batch_size, n_images)
            batch_images = data[i:batch_end]
            
            # Convert to torch tensor and reorder dimensions (N, H, W, C) -> (N, C, H, W)
            if batch_images.shape[-1] in [1, 3]:  # Channels last
                batch_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float()
            else:  # Assume channels first already
                batch_tensor = torch.from_numpy(batch_images).float()
            
            batch_tensor = batch_tensor.to(device)
            
            # Extract features
            feats = model.feature_extractor(batch_tensor)
            embeddings.append(feats.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)


def predict_distances(model, paired_dataset, batch_size=64, device=None):
    """
    Predict distances in latent space for a PairedDataset.

    Extracts embeddings for all unique images in the dataset and computes 
    pairwise Euclidean distances in the embedding space. This provides 
    predictions that can be compared against ground truth distances.

    :param model: Trained dGWOT model with a feature_extractor attribute.
    :type model: torch.nn.Module
    :param paired_dataset: Dataset containing paired images with known distances.
    :type paired_dataset: PairedDataset
    :param batch_size: Batch size for processing embeddings. Default is 64.
    :type batch_size: int, optional
    :param device: Device to run computation on. If None, uses model's current device.
    :type device: torch.device, optional
    :returns: Array of predicted distances of shape (len(paired_dataset),)
        corresponding to each pair in the dataset.
    :rtype: numpy.ndarray
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get all unique image indices from the dataset
    all_indices = set()
    for pair in paired_dataset.image_pairs:
        all_indices.update(pair)
    all_indices = sorted(list(all_indices))
    
    print(f"Extracting embeddings for {len(all_indices)} unique images...")
    
    # Create dataset for unique images
    unique_image_dataset = IndexedImageDataset(
        paired_dataset.image_dir, 
        all_indices, 
        transform=paired_dataset.transform
    )
    
    # Extract embeddings for all unique images
    embeddings = extract_embeddings(model, unique_image_dataset, batch_size=batch_size, device=device)
    
    # Create mapping from image index to embedding index
    index_to_embedding = {img_idx: emb_idx for emb_idx, img_idx in enumerate(all_indices)}
    
    # Compute distances for each pair in the dataset
    predicted_distances = []
    
    print(f"Computing distances for {len(paired_dataset.image_pairs)} pairs...")
    
    for pair in tqdm(paired_dataset.image_pairs, desc="Computing pairwise distances"):
        idx1, idx2 = pair
        
        # Get embedding indices
        emb_idx1 = index_to_embedding[idx1]
        emb_idx2 = index_to_embedding[idx2]
        
        # Get embeddings
        emb1 = embeddings[emb_idx1]
        emb2 = embeddings[emb_idx2]
        
        # Compute squared Euclidean distance (matching model's training objective)
        distance = np.sum((emb1 - emb2) ** 2)
        predicted_distances.append(distance)
    
    return np.array(predicted_distances)


def plot_distance_predictions(model, paired_dataset, batch_size=64, device=None, figsize=(8, 8), 
                             return_plot=False, title=None, alpha=0.6, s=20):
    """
    Plot predicted vs true distances for a PairedDataset.

    Creates a scatter plot comparing model predictions against ground truth 
    distances with a diagonal reference line and correlation metrics.

    :param model: Trained dGWOT model with a feature_extractor attribute.
    :type model: torch.nn.Module
    :param paired_dataset: Dataset containing paired images with known distances.
    :type paired_dataset: PairedDataset
    :param batch_size: Batch size for processing embeddings. Default is 64.
    :type batch_size: int, optional
    :param device: Device to run computation on. If None, uses model's current device.
    :type device: torch.device, optional
    :param figsize: Figure size as (width, height). Default is (8, 8).
    :type figsize: tuple, optional
    :param return_plot: Whether to return the matplotlib figure and axes objects. Default is False.
    :type return_plot: bool, optional
    :param title: Custom title for the plot. If None, uses default with correlation metrics.
    :type title: str, optional
    :param alpha: Transparency of scatter points. Default is 0.6.
    :type alpha: float, optional
    :param s: Size of scatter points. Default is 20.
    :type s: int, optional
    :returns: If return_plot is False: displays the plot and returns None.
        If return_plot is True: returns (fig, ax) matplotlib objects.
    :rtype: None or tuple
    """
    # Get predictions
    predicted_distances = predict_distances(model, paired_dataset, batch_size=batch_size, device=device)
    true_distances = np.array(paired_dataset.distances)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        sns.scatterplot(x=true_distances, y=predicted_distances, alpha=alpha, s=s, ax=ax)
        
        # Add diagonal reference line
        min_val = min(true_distances.min(), predicted_distances.min())
        max_val = max(true_distances.max(), predicted_distances.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect prediction')
        
        # Labels and title
        ax.set_xlabel('True Distances')
        ax.set_ylabel('Predicted Distances')
        
        if title:
            ax.set_title(title)
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if return_plot:
            return fig, ax
        else:
            plt.show()
            return None
            
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return None if not return_plot else (None, None)


def plot_reconstruction_comparison(model, paired_dataset, n_cells=5, device=None, figsize=None, seed=None):
    """
    Plot comparison of original mapped protein distributions vs model reconstructions.

    Randomly selects cell images from the dataset, shows the original mapped 
    protein distributions in the top row and their reconstructions from the model in 
    the bottom row.

    :param model: Trained dGWOT model with reconstruction capabilities.
    :type model: torch.nn.Module
    :param paired_dataset: Dataset containing paired images for reconstruction.
    :type paired_dataset: PairedDataset
    :param n_cells: Number of image pairs to display. Default is 5.
    :type n_cells: int, optional
    :param device: Device to run model on. If None, uses model's current device.
    :type device: torch.device, optional
    :param figsize: Figure size as (width, height). If None, automatically calculated 
        based on number of images.
    :type figsize: tuple, optional
    :param seed: Random seed for reproducible image selection. Default is None.
    :type seed: int, optional
    :returns: None. Displays the plot using matplotlib.
    :rtype: None
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get unique image indices from the dataset pairs
    all_image_indices = set()
    for pair in paired_dataset.image_pairs:
        all_image_indices.update(pair)
    all_image_indices = list(all_image_indices)
    
    # Randomly select unique image indices
    selected_image_indices = np.random.choice(all_image_indices, size=min(n_cells, len(all_image_indices)), replace=False)
    
    # Set up the plot
    if figsize is None:
        figsize = (3 * n_cells, 6)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, n_cells, figsize=figsize)
        if n_cells == 1:
            axes = axes.reshape(2, 1)
        
        model.eval()
        
        with torch.no_grad():
            for i, img_idx in enumerate(selected_image_indices):
                # Load the specific image directly by constructing the path
                img_path = os.path.join(paired_dataset.image_dir, f"cell_{img_idx}.npy")
                mapped_img_path = os.path.join(paired_dataset.mapped_image_dir, f"mapped_cell_{img_idx}.npy")
                
                # Load the numpy arrays
                image = np.load(img_path)
                mapped_image = np.load(mapped_img_path)
                
                # Apply transforms if they exist
                if paired_dataset.transform is not None:
                    image = paired_dataset.transform(image)
                    mapped_image = paired_dataset.transform(mapped_image)
                
                # Convert to tensor and add batch dimension
                if isinstance(image, np.ndarray):
                    # Convert from (H, W, C) to (1, C, H, W)
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                else:
                    # Already a tensor, just add batch dimension and ensure correct device
                    image_tensor = image.unsqueeze(0).to(device)
                
                # Get reconstruction from model
                _, reconstruction, _ = model(image_tensor, image_tensor)
                
                # Extract protein channel (channel 0)
                if isinstance(mapped_image, np.ndarray):
                    original_protein = mapped_image[:, :, 0]  # Protein channel
                else:
                    original_protein = mapped_image[0].cpu().numpy()  # Protein channel
                
                reconstructed_protein = reconstruction[0, 0].cpu().numpy()  # First batch, first (protein) channel
                
                # Plot original protein (top row)
                im1 = axes[0, i].imshow(original_protein, cmap='viridis')
                axes[0, i].set_title(f'Original Cell {img_idx}')
                axes[0, i].axis('off')
                
                # Plot reconstructed protein (bottom row)
                im2 = axes[1, i].imshow(reconstructed_protein, cmap='viridis')
                axes[1, i].set_title(f'Reconstructed Cell {img_idx}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
        return
    
    except Exception as e:
        print(f"Error creating plot: {e}")
        return
    

def generate_dataset_split_pairs(indices, n_pairs, proportions=None, seed=None):
    """
    Generate cell pairs for dGWOT dataset splits.
    
    Creates stratified cell pairs for deep learning model training. Supports both 
    random sampling across all cells and stratified sampling from predefined groups 
    to ensure balanced representation across train/validation/test splits.
    
    :param indices: List of available cell indices corresponding to processed cell images.
    :type indices: list or array-like
    :param n_pairs: N-length list specifying number of pairs to generate for each dataset split.
        Typically [n_train_pairs, n_val_pairs, n_test_pairs].
    :type n_pairs: list of int
    :param proportions: N-length list of proportions that sum to 1.0 for stratified sampling.
        If provided, cell indices are split into N groups according to these 
        proportions, and pairs are drawn only within each group. This ensures
        train/val/test sets use disjoint cell populations. If None, all pairs 
        are drawn randomly from all available cells. Default is None.
    :type proportions: list of float, optional
    :param seed: Random seed for reproducible dataset splits. Default is None.
    :type seed: int, optional
    :returns: N-length list where each element is a 2D array of shape (n_pairs, 2) 
        containing cell index pairs for each dataset split (train, val, test).
    :rtype: list of numpy.ndarray
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = np.array(indices)
    n_groups = len(n_pairs)  # Number of dataset splits (train, val, test)
    
    if proportions is not None:
        if len(proportions) != n_groups:
            raise ValueError(f"Length of proportions ({len(proportions)}) must match length of n_pairs ({n_groups})")
        
        if not np.isclose(sum(proportions), 1.0, rtol=1e-5):
            raise ValueError(f"Proportions must sum to 1.0, got {sum(proportions)}")
        
        # Split cell indices into disjoint groups for train/val/test
        np.random.shuffle(indices)  # Randomize cell order first
        n_total = len(indices)
        
        group_indices = []
        start_idx = 0
        
        for i, prop in enumerate(proportions[:-1]):  # Handle all but last group
            group_size = int(np.round(prop * n_total))
            end_idx = start_idx + group_size
            group_indices.append(indices[start_idx:end_idx])
            start_idx = end_idx
        
        # Last group gets remaining indices
        group_indices.append(indices[start_idx:])
        
        # Generate pairs within each disjoint cell group (train, val, test)
        paired_arrays = []
        for i, group_inds in enumerate(group_indices):
            n_pairs_for_group = n_pairs[i]
            if len(group_inds) < 2:
                raise ValueError(f"Dataset split {i} has only {len(group_inds)} cells, need at least 2 to generate pairs")
            
            pairs = get_random_pairs(group_inds, n_pairs_for_group)
            paired_arrays.append(pairs)
    
    else:
        # Generate all pairs randomly from all cells (overlapping populations)
        paired_arrays = []
        for n_pairs_for_group in n_pairs:
            pairs = get_random_pairs(indices, n_pairs_for_group)
            paired_arrays.append(pairs)
    
    return paired_arrays