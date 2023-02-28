import argparse
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dotmap import DotMap
from torch.utils.data import TensorDataset
from torchvision import models, datasets, transforms
from tqdm import tqdm
import os
from npt.model.npt_modules import MHSA
from npt.utils.train_utils import count_parameters
from npt.utils.dataUtil import getCudaPeakMem


def get_npt_config():
    return {
        'model_mix_heads': True,
        'model_sep_res_embed': True,
        'model_att_block_layer_norm': True,
        'model_rff_depth': 1,
        'model_att_score_norm': 'softmax',
        'model_pre_layer_norm': True,
        'viz_att_maps': False,
        'model_layer_norm_eps': 1e-12,
        'model_hidden_dropout_prob': 0.1,
        'model_att_score_dropout_prob': 0.1,
    }


def get_CIFAR10(root='./'):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_transforms_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_transform = transforms.Compose(train_transforms_list)
    train_dataset = datasets.CIFAR10(
        root + 'data/CIFAR10', train=True, transform=train_transform, download=True)

    test_transforms_list = [
        transforms.ToTensor(),
        normalize,
    ]

    test_transform = transforms.Compose(test_transforms_list)
    test_dataset = datasets.CIFAR10(
        root + 'data/CIFAR10', train=False, transform=test_transform, download=True)

    return input_size, num_classes, train_dataset, test_dataset


def get_MNIST(root="./"):
    input_size = 32
    num_classes = 10
    transform = transforms.Compose([
        # first, convert image to PyTorch tensor
        transforms.ToTensor(),
        # normalize inputs
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root + 'data/MNIST', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root + 'data/MNIST', train=False, transform=transform, download=True)

    return input_size, num_classes, train_dataset, test_dataset


class ResNet18Encoder(torch.nn.Module):
    """From
    https://github.com/y0ast/pytorch-snippets/blob/
    main/minimal_cifar/train_cifar.py,
    Due to Joost van Amersfoort (OATML Group)
    Minimal script to train ResNet18 to 94% accuracy on CIFAR-10.
    """

    def __init__(self, in_channels=3, encoding_dims=128, pretrained=False,
                 apply_log_softmax=True):
        super().__init__()
        if pretrained:
            # Need to give it the ImageNet dimensions
            self.resnet = models.resnet18(
                pretrained=pretrained, num_classes=1000)
        else:
            self.resnet = models.resnet18(
                pretrained=pretrained, num_classes=encoding_dims)

        self.resnet.conv1 = torch.nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()
        self.apply_log_softmax = apply_log_softmax

        # Replace last layer
        if pretrained:
            self.resnet.fc = nn.Linear(512, encoding_dims)

    def forward(self, x):
        x = self.resnet(x)
        if self.apply_log_softmax:
            x = F.log_softmax(x, dim=1)

        return x


"""NPT Variants. Perform ABD."""


class NPTLite(torch.nn.Module):
    def __init__(
            self, mhsa_args, stacking_depth=2, dim_hidden=128, dim_output=10):
        super().__init__()
        enc = []
        for _ in range(stacking_depth):
            enc.append(
                MHSA(dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden, mhsa_args))
        enc.append(nn.Linear(dim_hidden, dim_output))
        self.enc = nn.Sequential(*enc)

    def forward(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)

class SPINLite(torch.nn.Module):
    def __init__(
            self, mhsa_args, stacking_depth=2, dim_hidden=128, dim_output=10):
        super().__init__()
        enc = []
        for _ in range(stacking_depth):
            enc.append(
                MHSA(dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden, mhsa_args))
        enc.append(nn.Linear(dim_hidden, dim_output))
        self.enc = nn.Sequential(*enc)

    def forward(self, x):
        x = self.enc(x)
        x = torch.squeeze(x)
        return F.log_softmax(x, dim=1)        


class ResNetNPT(torch.nn.Module):
    def __init__(self, resnet_args, npt_args, npt_type, stacking_depth,
                 dim_hidden, train_label_masking_perc):
        super().__init__()
        resnet_args['apply_log_softmax'] = False
        self.embed = ResNet18Encoder(**resnet_args)

        # NPT should expect a one-hot label encoding + a mask dimension
        # We just assume we have dim_hidden 128, and we need a dim_hidden
        # that is evenly divided by the number of heads (8)
        # So we just embed target info to a 32 dim vector
        if train_label_masking_perc is not None:
            assert dim_hidden == 128, 'Hack, for now'
            dim_hidden += 32
            self.label_to_embedding_dim = nn.Linear(11, 32)

        dim_out = 10

        if npt_type == 'npt':
            self.npt = NPTLite(
                mhsa_args=DotMap(npt_args), stacking_depth=stacking_depth,
                dim_hidden=dim_hidden, dim_output=dim_out)
        else:
            raise NotImplementedError

    def forward(self, x):
        x_npt = self.embed(x)
        x_npt = torch.unsqueeze(x_npt, 0)
        return self.npt(x_npt)

class ResNetSPIN(torch.nn.Module):
    def __init__(self, resnet_args, npt_args, npt_type, stacking_depth,
                 dim_hidden, train_label_masking_perc):
        super().__init__()
        resnet_args['apply_log_softmax'] = False
        self.embed = ResNet18Encoder(**resnet_args)

        # NPT should expect a one-hot label encoding + a mask dimension
        # We just assume we have dim_hidden 128, and we need a dim_hidden
        # that is evenly divided by the number of heads (8)
        # So we just embed target info to a 32 dim vector
        if train_label_masking_perc is not None:
            assert dim_hidden == 128, 'Hack, for now'
            dim_hidden += 32
            self.label_to_embedding_dim = nn.Linear(11, 32)

        dim_out = 10

        if npt_type == 'spin':
            self.spin = SPINLite(
                mhsa_args=DotMap(npt_args), stacking_depth=stacking_depth,
                dim_hidden=dim_hidden, dim_output=dim_out)
        else:
            raise NotImplementedError

    def forward(self, x):
        x_npt = self.embed(x)
        x_npt = torch.unsqueeze(x_npt, 0)
        return self.spin(x_npt)

def train_with_label_masking(model, train_loader, optimizer, epoch,
                             exp_device, train_label_masking_perc):
    model.train()
    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.to(exp_device)
        target = target.to(exp_device)

        optimizer.zero_grad()

        embeddings = model.embed(data)

        # * Stochastic Label Masking *
        # Compute which indices we will mask
        n_targets = target.size(0)
        n_indices_to_mask = int(train_label_masking_perc * n_targets)
        indices_to_mask = np.random.choice(
            n_targets, n_indices_to_mask, replace=False)

        # We will randomize the label of 10% of these indices,
        # and actually mask out (set to 0) 90% as in BERT.
        n_random_indices = int(len(indices_to_mask) * 0.1)
        random_indices = indices_to_mask[:n_random_indices]
        mask_indices = indices_to_mask[n_random_indices:]

        # Prepare targets
        targets_to_input = target.clone()

        # One-hot encode targets
        targets_to_input = F.one_hot(targets_to_input)

        # Randomize the random indices
        n_classes = targets_to_input.size(1)
        new_class_indices = np.random.choice(
            n_classes, size=len(random_indices))

        for random_index, new_class_index in zip(
                random_indices, new_class_indices):
            targets_to_input[random_index, :] = 0
            targets_to_input[random_index, new_class_index] = 1

        # Mask out the mask indices
        for mask_index in mask_indices:
            targets_to_input[mask_index, :] = 0

        # Prepare mask indicators
        mask_indicators = torch.zeros(n_targets)
        mask_indicators[mask_indices] = 1
        mask_indicators = torch.unsqueeze(
            mask_indicators, dim=1).to(exp_device)

        # Append them to the end of the data
        target_embeddings = torch.cat(
            (targets_to_input, mask_indicators), dim=1)
        target_embeddings = model.label_to_embedding_dim(target_embeddings)
        embeddings = torch.cat(
            (embeddings, target_embeddings), dim=1)

        # Carry on with NPT!
        embeddings = torch.unsqueeze(embeddings, 0)
        prediction = model.npt(embeddings)

        # Eval the loss
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")
    return {'train_loss': avg_loss}


def train(model, train_loader, optimizer, epoch, exp_device):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.to(exp_device)
        target = target.to(exp_device)
        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")
    return {'train_loss': avg_loss}


def corrupt_data(input_arr, row_index, debug_ablate_shuffle=False,
                 is_image_data=True):
    corrupted_arr = input_arr.clone()

    if is_image_data:
        # Currently in shape (N, C, H, W)
        # Reshape to (N, D)
        n_channels, height, width = (
            corrupted_arr.size(1), corrupted_arr.size(2),
            corrupted_arr.size(3))
        corrupted_arr = corrupted_arr.reshape(corrupted_arr.size(0), -1)

    n_rows, n_cols = corrupted_arr.size(0), corrupted_arr.size(1)

    # Row indices to shuffle -- exclude the given row_index
    row_indices = list(set(range(n_rows)) - {row_index})

    # Shuffle all rows other than our selected one, row_index
    # Perform an independent permutation for each column so the row info
    # is destroyed (otherwise, our row-equivariant model won't have an
    # issue with permuted rows).
    for col in range(n_cols):
        # Test -- if we ablate shuffle, do not swap around elements
        if not debug_ablate_shuffle:
            shuffled_row_indices = np.random.permutation(row_indices)

            # Shuffle masked_tensors, which our model sees at input.
            # Don't need to shuffle data_arrs, because the row at which
            # we evaluate loss will be in the same place.
            corrupted_arr[:, col][row_indices] = corrupted_arr[
                                                 :, col][shuffled_row_indices]

    if is_image_data:
        return corrupted_arr.reshape(
            corrupted_arr.size(0), n_channels, height, width)
    else:
        return corrupted_arr


def get_test_time_label_embeddings(
        model, train_labels, test_labels, exp_device):
    """
    Get one-hot encodings of training and test labels, with all test labels
    masked and training labels available.
    """
    n_test_examples = len(test_labels)

    # One-hot embed train labels
    train_labels = F.one_hot(train_labels)

    n_train_examples, n_classes = train_labels.size()

    # All test label info is zeroed out
    test_zeros = torch.zeros((test_labels.size(0), n_classes))

    # Masks should indicate that all test rows are masked
    masks = torch.zeros(n_train_examples + n_test_examples)
    masks[-n_test_examples:] = 1

    # Concat labels and masks
    one_hot_labels = torch.cat((train_labels, test_zeros), dim=0)
    masks = torch.unsqueeze(masks, 1)
    one_hot_labels = torch.cat((
        one_hot_labels, masks), dim=1).to(exp_device)
    label_embeddings = model.label_to_embedding_dim(one_hot_labels)
    return label_embeddings


def npt_test(model, train_at_test_loader, test_loader, exp_device,
             provide_train_labels):
    """Provide train labels if we are doing stochastic label masking"""
    model.eval()

    loss = 0
    correct = 0

    for ((train_data, train_labels),
         (test_data, test_labels)) in zip(train_at_test_loader, test_loader):
        with torch.no_grad():
            data = torch.cat([train_data, test_data], dim=0)
            data = data.to(exp_device)
            test_labels = test_labels.to(exp_device)
            n_test_examples = len(test_labels)

            if provide_train_labels:
                # Get test time embeddings of the labels (test are masked)
                label_embeddings = get_test_time_label_embeddings(
                    model, train_labels, test_labels, exp_device)

                # Forward pass: obtain feature embeddings
                embeddings = model.embed(data)

                # Concatenate feature and label embeddings
                embeddings = torch.cat((embeddings, label_embeddings), dim=1)
                embeddings = torch.unsqueeze(embeddings, dim=0)
                prediction = model.npt(embeddings)
            else:
                prediction = model(data)

            prediction = prediction[-n_test_examples:, :]
            loss += F.nll_loss(prediction, test_labels, reduction="sum")
            prediction = prediction.max(1)[1]
            correct += prediction.eq(
                test_labels.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return {
        'test_loss': loss,
        'test_accuracy': percentage_correct
    }


def no_npt_test(model, test_loader, exp_device):
    """Eval a vanilla ResNet."""
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.to(exp_device)
            target = target.to(exp_device)
            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")
            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return {
        'test_loss': loss,
        'test_accuracy': percentage_correct
    }


def test_model_row_corruptions(
        model, train_at_test_loader, test_loader, test_batch_size,
        exp_device, provide_train_labels, is_image_data):
    model.eval()

    loss = 0
    correct = 0

    for i, (
            (train_data, train_labels), (test_data, test_labels)
    ) in enumerate(zip(train_at_test_loader, test_loader)):
        with torch.no_grad():
            input_data = torch.cat([train_data, test_data], dim=0)

            if provide_train_labels:
                label_embeddings = get_test_time_label_embeddings(
                    model, train_labels, test_labels, exp_device)

            test_labels = test_labels.to(exp_device)

            for test_row_index in range(
                    train_data.size(0),
                    train_data.size(0) + test_data.size(0)):
                target_index = test_row_index - train_data.size(0)
                data = corrupt_data(
                    input_arr=input_data, row_index=test_row_index,
                    is_image_data=is_image_data)

                data = data.to(exp_device)

                if provide_train_labels:
                    # Corrupt label embeddings
                    corrupted_label_embeddings = corrupt_data(
                        input_arr=label_embeddings, row_index=test_row_index,
                        is_image_data=False)

                    # Forward pass: obtain feature embeddings
                    embeddings = model.embed(data)

                    # Concatenate feature and corrupted label embeddings
                    embeddings = torch.cat(
                        (embeddings, corrupted_label_embeddings), dim=1)
                    embeddings = torch.unsqueeze(embeddings, dim=0)
                    prediction = model.npt(embeddings)
                else:
                    prediction = model(data)

                prediction = torch.unsqueeze(prediction[test_row_index, :], 0)
                current_target = torch.unsqueeze(test_labels[target_index], 0)
                loss += F.nll_loss(prediction, current_target, reduction="sum")
                prediction = prediction.max(1)[1]
                correct += prediction.eq(
                    current_target.view_as(prediction)).sum().item()

        if i % 100 == 0:
            print(f'Completed {i}/{len(test_loader)} batches.')
            print(
                f'Accuracy: {100.0 * correct / ((i + 1) * test_data.size(0))}')

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return {
        'test_loss': loss,
        'test_accuracy': percentage_correct
    }


def str2bool(v):
    """https://stackoverflow.com/questions/15008758/
    parsing-boolean-values-with-argparse/36031646"""
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def get_checkpoint_path(args):
    if args.model_checkpoint_key is None:
        checkpoint_key = args.npt_type
    else:
        checkpoint_key = args.model_checkpoint_key

    print(f'using checkpoint key {checkpoint_key}')

    checkpoint_dir = os.path.join(
        args.checkpoint_dir, f'{checkpoint_key}')
    file_path = os.path.join(
        checkpoint_dir, f'model_epoch_{args.epochs}.pt')
    return checkpoint_dir, file_path


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument("--use_wandb", type='bool', default=True,
                        help="if enabled, used wandb")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="learning rate (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument(
        "--eval_checkpoint", type='bool', default=False,
        help='If True, load from checkpoint and evaluate on test set.')
    parser.add_argument("--checkpoint_dir", type=str,
                        default='./checkpoints/')
    parser.add_argument("--data_set", type=str, default='cifar-10')
    parser.add_argument("--exp_use_cuda", type='bool', default=True)
    parser.add_argument("--exp_device", type=str, default=None)

    # * New Arguments *

    # Don't need tuning
    parser.add_argument(
        '--att_expansion',
        default=4,
        type=int,
        help='4 times expansion factor')
    parser.add_argument("--data_loader_nprocs", type=int, default=4,
                        help='Torch dataloader number of processes.')
    parser.add_argument("--model_checkpoint_key", type=str, default=None,
                        help='Subdir where model checkpoints are stored.')
    parser.add_argument(
        "--checkpoint_override", type='bool', default=False,
        help="If True, allows row corruption evaluation without caching")

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--train_at_test_batch_size", type=int, default=40)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument(
        "--dim_hidden", type=int, default=128,
        help='Num hidden dims that the ResNet18 outputs, '
             'and that we attend between in NPT.')
    parser.add_argument('--track_model_frequency', type=int, default=None,
                        help='if not none, set frequency of model tracking')
    parser.add_argument('--model_num_heads', type=int, default=8,
                        help='Number of MHSA heads.')
    parser.add_argument("--use_pretrained_resnet", type='bool', default=False,
                        help='If True, use a pretrained ResNet.')
    parser.add_argument("--stacking_depth", type=int, default=8,
                        help="depth of the NPT")
    parser.add_argument(
        "--test_row_corruptions", type='bool', default=False,
        help="If enabled, run as many forward passes as there are test "
             "examples, corrupting all other elements.")
    parser.add_argument("--npt_type", type=str, default=None,
                        help="If None, don't use NPT. Otherwise, select in set "
                             "{npt, spin}")
    parser.add_argument(
        "--train_label_masking_perc", type=float, default=None,
        help="If specified, perform label masking at train time (i.e., "
             "--train_label_masking_perc percent of labels will be masked, "
             "and the rest will be visible.")
    parser.add_argument('--exp_multistep_milestones', type=str, default=None,
                        help='Epoch milestones for multistep LR scheduler.')

    # Wandb args
    parser.add_argument('--project', type=str, default='npt-debug',
                        help='Wandb project name.')
    parser.add_argument('--wandb_dir', type=str, default='./',
                        help='Directory to which wandb logs.')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Give experiment a name.')
    parser.add_argument('--exp_group', type=str, default=None,
                        help='Give experiment a group name.')

    args = parser.parse_args()

    if args.exp_use_cuda and torch.cuda.is_available():
        if args.exp_device is not None:
            print(f'Running model with CUDA on device {args.exp_device}.')
            exp_device = args.exp_device
        else:
            print(f'Running model with CUDA')
            exp_device = 'cuda:0'
    else:
        print('Running model on CPU.')
        exp_device = 'cpu'

    if args.checkpoint_override:
        if args.eval_checkpoint:
            raise NotImplementedError(
                'Cannot combine a cache with a cache override')
    else:
        if args.test_row_corruptions and not args.eval_checkpoint:
            raise RuntimeError(
                'Error - testing row corruptions requires used of \
                a cached model unless using checkpoint override')

    args.exp_device = exp_device
    print(args)
    torch.manual_seed(args.seed)

    # Wandb Setup

    pathlib.Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb_args = dict(
        project=args.project,
        # entity="rr568",
        dir=args.wandb_dir,
        reinit=True,
        name=args.exp_name,
        group=args.exp_group)
    wandb_run = wandb.init(
        **wandb_args, mode='online' if args.use_wandb else 'disabled')
    wandb.config.update(args, allow_val_change=True)
    # c = wandb.config
    args.cv_index = 0

    # Dataset and Loader Setup
    is_image_data = True

    if args.data_set == 'cifar-10':
        dataset_fn = get_CIFAR10
    elif args.data_set == 'mnist':
        dataset_fn = get_MNIST
    else:
        raise NotImplementedError

    input_size, num_classes, train_dataset, test_dataset = dataset_fn()

    if args.train_batch_size == -1:
        print('Train batch size set to -1, overriding to use full batch '
              'training and evaluation throughout.')
        assert args.train_at_test_batch_size == -1
        assert args.test_batch_size == -1
        full_batch = True
    else:
        full_batch = False

    train_batch_size = (
        len(train_dataset) if full_batch else args.train_batch_size)
    train_at_test_batch_size = (
        len(train_dataset) if full_batch else args.train_at_test_batch_size)
    test_batch_size = (
        len(test_dataset) if full_batch else args.test_batch_size)

    kwargs = {"num_workers": args.data_loader_nprocs, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size,
        shuffle=True, **kwargs)
    train_at_test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_at_test_batch_size,
        shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size,
        shuffle=True, **kwargs)

    # Model Args Setup

    if is_image_data:
        resnet_args = {
            'in_channels': 3,
            'encoding_dims': args.dim_hidden,
            'pretrained': args.use_pretrained_resnet
        }
        print('ResNet args:', resnet_args)
    npt_args = get_npt_config()
    npt_args['input_size'], npt_args['num_classes'] = input_size, num_classes
    npt_args['model_num_heads'] = args.model_num_heads
    npt_args['max_batch_size'] = max(
        train_loader.batch_size,
        train_at_test_loader.batch_size,
        test_loader.batch_size)
    npt_args['dim_out'] = num_classes
    npt_args['exp_device'] = args.exp_device

    stacking_depth = args.stacking_depth
    npt_type = args.npt_type
    model_str = 'ResNet18' if is_image_data else ''
    if args.npt_type is not None:
        model_str += f' + {args.npt_type} ({args.stacking_depth} Layers)'

    # Model Setup

    if npt_type is None:
        model = ResNet18Encoder(
            in_channels=3, encoding_dims=10,
            pretrained=args.use_pretrained_resnet,
            apply_log_softmax=True)
    elif npt_type == 'spin':
        model = ResNetSPIN(
            resnet_args=resnet_args, npt_args=npt_args, npt_type=npt_type,
            stacking_depth=stacking_depth, dim_hidden=args.dim_hidden,
            train_label_masking_perc=args.train_label_masking_perc
        )
    else:
        model = ResNetNPT(
            resnet_args=resnet_args, npt_args=npt_args, npt_type=npt_type,
            stacking_depth=stacking_depth, dim_hidden=args.dim_hidden,
            train_label_masking_perc=args.train_label_masking_perc)

    print(f'Model has {count_parameters(model)} parameters.')
    wandb.run.summary.update({'params': count_parameters(model)})
    
    # Run Evaluation (from Checkpoint) or Training

    if args.eval_checkpoint:
        checkpoint_dir, file_path = get_checkpoint_path(args)
        model.load_state_dict(torch.load(file_path))
        model = model.to(exp_device)
        if args.track_model_frequency is not None:
            wandb.watch(model, log_freq=args.track_model_frequency)
        print(f'Successfully loaded checkpoint from path {file_path}.')

        if args.test_row_corruptions:
            assert npt_type is not None, (
                'Can only test row corruptions w/ NPT.')
            print('Testing row corruptions.')
            test_dict = test_model_row_corruptions(
                model, train_at_test_loader, test_loader,
                test_batch_size=args.test_batch_size, exp_device=exp_device,
                provide_train_labels=(
                        args.train_label_masking_perc is not None),
                is_image_data=is_image_data)
        else:
            print(f'Evaluating checkpointed {model_str} for '
                  f'{args.epochs} epochs.')

            if npt_type is None:
                test_dict = no_npt_test(model, test_loader, exp_device)
            else:
                test_dict = npt_test(
                    model, train_at_test_loader, test_loader, exp_device,
                    provide_train_labels=(
                            args.train_label_masking_perc is not None))

        wandb.log(test_dict, step=1)

    else:
        model = model.to(exp_device)
        if args.track_model_frequency is not None:
            wandb.watch(model, log_freq=args.track_model_frequency)
        if args.exp_multistep_milestones is None:
            milestones = [25, 40]
        else:
            milestones = list(args.exp_multistep_milestones)
            print(f'Using milestones {milestones}.')

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

        print(f'Training {model_str} for {args.epochs} epochs.')
        wandb.run.summary.update({'GPU_Mem': getCudaPeakMem()})
        for epoch in range(1, args.epochs + 1):
            print(f'LR: {scheduler.get_last_lr()}')

            if args.train_label_masking_perc is None:
                train_dict = train(
                    model, train_loader, optimizer, epoch, exp_device)
            else:
                train_dict = train_with_label_masking(
                    model, train_loader, optimizer, epoch, exp_device,
                    args.train_label_masking_perc)

            # Log to wandb
            results_dict = train_dict.copy()

            # if not args.checkpoint_override:
            if npt_type is None:
                test_dict = no_npt_test(model, test_loader, exp_device)
            else:
                test_dict = npt_test(
                    model, train_at_test_loader, test_loader, exp_device,
                    provide_train_labels=(
                            args.train_label_masking_perc is not None))

            # Log to wandb
            results_dict.update(test_dict)

            wandb.log(results_dict, step=epoch)
            scheduler.step()

        if args.checkpoint_override:
            print('using cache override to test model')
            # if args.test_row_corruptions:
            assert npt_type is not None, (
                'Can only test row corruptions w/ NPT.')
            print('Testing row corruptions.')
            test_dict = test_model_row_corruptions(
                model, train_at_test_loader, test_loader,
                test_batch_size=args.test_batch_size,
                exp_device=exp_device, provide_train_labels=(
                        args.train_label_masking_perc is not None),
                is_image_data=is_image_data)
            # else:
            print('testing trained model with std evaluation')
            if npt_type is None:
                test_dict = no_npt_test(model, test_loader, exp_device)
            else:
                test_dict = npt_test(
                    model, train_at_test_loader, test_loader,
                    exp_device, provide_train_labels=(
                            args.train_label_masking_perc is not None))

            # wandb.log(test_dict, step=epoch+1)

        checkpoint_dir, file_path = get_checkpoint_path(args)
        print(
            f'Saving checkpoint at epoch {args.epochs} '
            f'to path {file_path}.')
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(file_path))
        print('Successfully saved checkpoint.')

    wandb_run.finish()


if __name__ == "__main__":
    main()
