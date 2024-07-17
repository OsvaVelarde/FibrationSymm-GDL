import torch.nn as nn
import torch.nn.functional as F
import torch

# ---------------------------------------------------------------

class multilayer_perceptron(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, bias):
        super(multilayer_perceptron, self).__init__()
        self.in_size = in_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_sizes[0], bias=bias))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=bias))
        self.last_layer = nn.Linear(hidden_sizes[-1], out_size, bias=bias)        

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_dict = {}
        out = x.view(-1, self.in_size)
        for id_layer, layer in enumerate(self.layers):
            out = self.relu(layer(out))
            out_dict['h'+str(id_layer+1)] = out
            #out = self.dropout(out)

        out = self.last_layer(out)
        out_dict['h'+str(id_layer+2)] = out

        return out_dict
    
# ---------------------------------------------------------------

class multilayer_fibers(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, bias, list_sizes):
        super(multilayer_fibers, self).__init__()
        self.in_size = in_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_sizes[0], bias=bias))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=bias))
        self.last_layer = nn.Linear(hidden_sizes[-1], out_size, bias=bias)        

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        sizes_fibers =  [[1 for i in range(in_size)]] + list_sizes + [[1 for i in range(out_size)]]
        self.sizes_fibers = nn.ParameterList([nn.Parameter(torch.tensor(x,dtype=torch.float32),requires_grad=False) for x in sizes_fibers])

    def forward(self, x):
        out_dict = {}
        out = x.view(-1, self.in_size)
        for id_layer, layer in enumerate(self.layers):
            out = self.relu(layer(out))
            out_dict['h'+str(id_layer+1)] = out
            #out = self.dropout(out)

        out = self.last_layer(out)
        out_dict['h'+str(id_layer+2)] = out

        return out_dict
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------
#'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, ch_layers):
        super(VGG, self).__init__()
        self.features = self._make_layers(ch_layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)          # N x Ch x H x W
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# ---------------------------------------------------------------

class VGG_fibers(nn.Module):
    def __init__(self,in_size, out_size, hidden,list_sizes):
        super(VGG_fibers, self).__init__()
        self.features = self._make_layers(in_size,hidden)
        self.classifier = nn.Linear(hidden[-2], out_size)

        sizes_fibers =  [[1 for i in range(in_size)]] + list_sizes + [[1 for i in range(out_size)]]
        self.sizes_fibers = nn.ParameterList([nn.Parameter(torch.tensor(x,dtype=torch.float32),requires_grad=False) for x in sizes_fibers])

    def forward(self, x):
        out = self.features(x)
        h_features = out.view(out.size(0), -1)
        h_class = self.classifier(h_features)
        return {'h1':h_features,'h2':h_class}

    def _make_layers(self,in_channels,cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# ---------------------------------------------------------------

_LIST_MODELS = {'MLP': multilayer_perceptron,'MLP_F': multilayer_fibers, 
                'CNN':VGG, 'CNN_F': VGG_fibers}