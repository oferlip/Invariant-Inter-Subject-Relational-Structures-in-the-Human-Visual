import torch
import os
from datetime import datetime
from torch import optim
import torch.nn as nn

def get_model_path(model_directory, model_name):
    return os.path.join(model_directory, model_name)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def train_network(net,
                  number_of_epoch,
                  trainloader,
                  optimizer=None,
                  criterion=nn.MSELoss(),
                  path=None,
                  save_every_5_epoches=False,
                  testloader=None,
                  custom_loss=None):

    if optimizer == None:
        optimizer = optim.Adam(net.parameters(), lr=1e-3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    # print(device)

    print_loss_after_instances = 10000
    if path == None and save_every_5_epoches:
        print("path is null, can not save every 5 epoches")

    for epoch in range(number_of_epoch):  # loop over the dataset multiple times

        if epoch % 5 == 0 and epoch > 0:
            if save_every_5_epoches and path != None:
                print("saving network on epoch {}".format(epoch))
                save_network(path, net)
                print("finished saving network on epoch {}".format(epoch))

            if testloader != None:
                test_model(net, testloader)

        running_loss = 0.0
        time = datetime.now()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if not custom_loss is None:
                loss = custom_loss(inputs, labels, outputs)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_loss_after_instances == print_loss_after_instances - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_loss_after_instances))
                print("it took {}".format(datetime.now() - time))
                time = datetime.now()
                running_loss = 0.0

    # print('Finished Training')


def test_model(net, testloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    net.train()


def get_models_hidden_layers(net, dataset, amount_to_remain):
    # save_output = SaveOutput()
    hook_handles = []
    # for i, layer in enumerate(net.modules()):
    #     if i > 0 and net.is_layer_for_cka(layer):
    #         handle = layer.register_forward_hook(save_output)
    #         hook_handles.append(handle)

    result = []
    net.eval()
    print("starting get_models_hidden_layers")
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= amount_to_remain:
                break

            # save_output.clear()
            image, label = data

            outputs = net(image)

            for j, hidden_value in enumerate(net.last_instance_hidden):
                if len(result) == j:
                    result.append([])
                result[j].append(torch.reshape(hidden_value, (-1,)))

    for i in range(len(result)):
        result[i] = torch.stack(result[i])

    print("finished get_models_hidden_layers")
    return result


def save_network(path, net):
    torch.save(net.state_dict(), path)


def load_network(path, net):
    net.load_state_dict(torch.load(path))


def get_hidden_layer(net, layer, image):
    with torch.no_grad():
        net(image)
        hidden = net.get_last_instance_hidden()
        return hidden[layer]
