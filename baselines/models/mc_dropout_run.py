import torch
import tqdm
import math
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader
from baselines.models.mc_dropout_modules import MLP, MLPClassificationModel, MLPRegressionModel
import pdb
import numpy as np

def main(c, dataset):
    tune_model(c, dataset, hyper_dict=None)


def build_dataloaders(dataset, batch_size):
    data_dict = dataset.cv_dataset
    D = data_dict['D']
    cat_target_cols, num_target_cols = (
        data_dict['cat_target_cols'], data_dict['num_target_cols'])
    target_cols = list(sorted(cat_target_cols + num_target_cols))
    non_target_cols = sorted(
        list(set(range(D)) - set(target_cols)))
    train_indices, val_indices, test_indices = (
        tuple(data_dict['new_train_val_test_indices']))
    data_arrs = data_dict['data_arrs']

    X = []
    y = None
    sigma_target_col=None
    for i, col in enumerate(data_arrs):
        if i in non_target_cols:
            col = col[:, :-1]
            X.append(torch.as_tensor(col))
        else:
            col = col[:, :-1]
            y = torch.as_tensor(col)

    X = torch.cat(X, dim=-1)
    X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]

    if y.shape[1] > 1:
        dataset_is_classification = True
        num_classes = y.shape[1]
        y = torch.argmax(y.long(), dim=1)
    else:
        dataset_is_classification = False
        num_classes = None
        sigma_target_col = data_dict['sigmas'][num_target_cols[-1]]

    y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
    train_dataset, val_dataset, test_dataset = (
        TensorDataset(X_train, y_train),
        TensorDataset(X_val, y_val),
        TensorDataset(X_test, y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return (
        (train_loader, val_loader, test_loader), X.shape[1],
        dataset_is_classification, num_classes, X_train, sigma_target_col)

def tune_model(c, dataset, hyper_dict):
    batch_size = c.exp_batch_size
    dataloaders, input_dims, is_classification, num_classes, X_train, sigma_target_col = (
        build_dataloaders(dataset=dataset, batch_size=batch_size))
    train_loader, val_loader, test_loader = dataloaders

    # Define some hypers here

    # This is the output of the feature extractor, which is then
    # transformed to grid space (in classification) or
    # is the size of the inducing points (regression)
    # We init the inducing points by projecting a random selection of
    # training points, and running KMeans on that
    num_features = 10
    hidden_layers = [256]
    dropout_prob = 0.4
    n_epochs = 10000
    lr = 0.001
    feature_extractor_weight_decay = 1e-4
    scheduler__milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    scheduler__gamma = 0.1
 
    # Define some hypers here

    feature_extractor = MLP(
        input_size=input_dims, hidden_layer_sizes=hidden_layers,
        output_size=num_features, dropout_prob=dropout_prob)

    if is_classification:
        model = MLPClassificationModel(feature_extractor, num_dim=num_features)
        loss_criteria=torch.nn.CrossEntropyLoss()
    else:
        model = MLPRegressionModel(
            feature_extractor, input_size=num_features, hidden_layer_sizes=hidden_layers,
            output_size=1, dropout_prob=dropout_prob)
        loss_criteria=torch.nn.MSELoss()

    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()

    # Train loop
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(),
         'weight_decay': feature_extractor_weight_decay},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    scheduler = MultiStepLR(
        optimizer, milestones=scheduler__milestones, gamma=scheduler__gamma)

    def train(epoch):
        model.train()
        loss_ls=[]
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            data = data.reshape(data.size(0), -1)

            optimizer.zero_grad()
            output = model(data.float())
            loss = loss_criteria(output, target)
            loss.backward()
            loss_ls.append(loss.item())
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        print(f"loss:{sum(loss_ls)/len(loss_ls)} in epoch:{epoch}")
    mc_samples=10
    is_mcdropout=True
    def activate_mc_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def test(data_loader, mode):
        model.eval()
        if is_classification:
            correct = 0
        else:
            mse = 0
            num_batches = 0

        with torch.no_grad():
            for data, target in data_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                data = data.reshape(data.size(0), -1)

                if is_classification:
                    output = model(data.float())
                    pred = output.probs.mean(0)
                    pred = pred.argmax(-1)  # Taking the mean over all of the sample we've drawn
                    correct += pred.eq(target.view_as(pred)).cpu().sum()
                else:
                    if is_mcdropout:
                        activate_mc_dropout(model)
                    preds = model(data.float())
                    mse += torch.mean((preds - target) ** 2).item()
                    num_batches += 1

        if is_classification:
            print('{} set: Accuracy: {}/{} ({}%)'.format(
                mode,
                correct, len(test_loader.dataset), 100. * correct / float(len(data_loader.dataset))
            ))
        else:
            # print('{} set: MSE: {}'.format(mode, mse / num_batches))
            print('{} set: RMSE: {}'.format(mode, math.sqrt(mse*(sigma_target_col**2) / num_batches)))
        if is_mcdropout and (not is_classification):
            return math.sqrt(mse*(sigma_target_col**2)/ num_batches)

    for epoch in range(1, n_epochs + 1):
            train(epoch)
            test(val_loader, mode='Val')
            if is_mcdropout and (not is_classification):
                rmse_unstd=[]
                for _ in range(mc_samples):
                    rmse_unstd.append(test(data_loader=test_loader, mode='Test'))
                print(f"mean of rmse for {mc_samples} in epoch{epoch}: {np.array(rmse_unstd).mean()} and std:{np.array(rmse_unstd).std()}")
            else:
                test(data_loader=test_loader, mode='Test')
            scheduler.step()
            state_dict = model.state_dict()
            torch.save({'model': state_dict}, 'mc_dropout_checkpoint.dat')
            

