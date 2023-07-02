import time

import torch

# from mac_config import device


def train(epoch, model, train_loader, criterion, optimizer, scheduler,_device):

    global loss
    model.train()
    correct = 0
    start = time.time()

    for data, targets in train_loader:
        input, target = data.to(_device), targets.to(_device)
        output = model(input)
        # print('output = ', output)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    scheduler.step()
    accuracy = 100. * correct / len(train_loader.dataset)

    end = time.time()
    print(f'finish epoch {epoch + 1} training in {end - start:.4f} s')

    return loss.item(), accuracy


def test(model, test_loader, criterion, epoch, _device):
    model.eval()
    counter = 0
    correct = 0
    valid_running_loss = 0.0
    start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            counter += 1
            data, target = data.to(_device), target.to(_device)
            output = model(data)
            _loss = criterion(output, target)
            valid_running_loss += _loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_loss = valid_running_loss / counter
    end = time.time()
    print(f'finish epoch {epoch + 1} testing in {end - start:.4f} s')

    return test_loss, test_accuracy
