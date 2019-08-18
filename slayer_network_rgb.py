# Original work Copyright (c) 2019 Christoph Hofer
# Modified work Copyright (c) 2019 Wolf Byttner
#
# This file is part of the code implementing the thesis
# "Classifying RGB Images with multi-colour Persistent Homology".
#
#     This file is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Lesser General Public License as published
#     by the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This file is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public License
#     along with this file.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import numpy as np

import sys
import os

import multiprocessing

from sklearn.preprocessing.label import LabelEncoder
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from rotated_persistence_diagrams_rgb import colours, get_folder_string
from birds_data import images, labels, categories, training_data_labels
from chofer_nips2017.src.sharedCode.provider import Provider
from chofer_nips2017.src.sharedCode.experiments import \
    UpperDiagonalThresholdedLogTransform, \
    pers_dgm_center_init, SLayerPHT, \
    PersistenceDiagramProviderCollate
from sklearn.model_selection import StratifiedShuffleSplit
from rotated_persistence_diagrams_rgb import rotate_all_persistence_diagrams

sys.path.append(os.path.join(os.path.dirname(__file__),
                             "chofer_nips2017/chofer_torchex"))

import chofer_torchex.utils.trainer as tr
from chofer_torchex.utils.trainer.plugins import *


def _parameters():
    return \
        {
            'colour_mode': 'rgb',
            'data_path': None,
            'epochs': 100,
            'momentum': 0.9,
            'lr_start': 0.05,
            'lr_ep_step': 1,
            'lr_adaption': 0.5,
            'test_ratio': 0.1,
            'batch_size': 8,
            'directions': 32,
            'resampled_size': (16, 16),
            'sat_key': 'tmp.csv',
            'cuda': False
        }


def serialise_params(params):
    serial = ""
    for key, val in params.items():
        serial += "_{}_{}".format(key, val)
    return serial


def BasicLayer(in_channels, out_channels):
    layer = nn.Sequential()
    layer.add_module('conv', nn.Conv2d(in_channels, out_channels, 3))
    layer.add_module('batch_norm', nn.BatchNorm2d(out_channels))
    layer.add_module('relu', nn.ReLU())
    return layer


class SLayerRgbNN(torch.nn.Module):
    def __init__(self, subscripted_views, directions):
        super(SLayerRgbNN, self).__init__()
        self.subscripted_views = subscripted_views

        print(subscripted_views)
        n_elements = 75
        n_filters = directions
        stage_2_out = 25
        n_neighbor_directions = 1

        self.transform = UpperDiagonalThresholdedLogTransform(0.1)

        self.pht_sl = SLayerPHT(len(subscripted_views),
                                n_elements,
                                2,
                                n_neighbor_directions=n_neighbor_directions,
                                center_init=self.transform(
                                        pers_dgm_center_init(n_elements)),
                                sharpness_init=torch.ones(n_elements, 2) * 4)

        self.stage_1 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(1 + 2 * n_neighbor_directions,
                                               n_filters//2, 1, bias=False))
            seq.add_module('batch_norm_1', nn.BatchNorm1d(n_filters//2))
            seq.add_module('relu_1', nn.ReLU())

            self.stage_1.append(seq)
            self.add_module('stage_1_{}'.format(i), seq)

        self.stage_3 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_1', nn.Conv1d(n_filters//2, n_filters//2, 1, bias=False))
            seq.add_module('batch_norm_1', nn.BatchNorm1d(n_filters//2))
            seq.add_module('relu_1', nn.ReLU())

            seq.add_module('conv_2', nn.Conv1d(n_filters//2, n_filters//2, 1, bias=False))
            seq.add_module('batch_norm_2', nn.BatchNorm1d(n_filters//2))
            seq.add_module('relu_2', nn.ReLU())

            self.stage_3.append(seq)
            self.add_module('stage_3_{}'.format(i), seq)

        self.stage_4 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('conv_3', nn.Conv1d(n_filters//2, n_filters, 1, bias=False))
            seq.add_module('batch_norm_3', nn.BatchNorm1d(n_filters))
            seq.add_module('relu_3', nn.ReLU())

            #seq.add_module('conv_4', nn.Conv1d(n_filters, n_filters, 1, bias=False))
            #seq.add_module('batch_norm_4', nn.BatchNorm1d(n_filters))
            #seq.add_module('relu_4', nn.ReLU())

            self.stage_4.append(seq)
            self.add_module('stage_4_{}'.format(i), seq)

        self.stage_2 = []
        for i in range(len(subscripted_views)):
            seq = nn.Sequential()
            seq.add_module('batch_norm_1', nn.BatchNorm1d(n_elements))
            seq.add_module('Dropout_1', nn.Dropout(0.25))
            seq.add_module('linear_1', nn.Linear(n_elements, n_elements//2))
            seq.add_module('relu_1', nn.ReLU())
            seq.add_module('batch_norm_2', nn.BatchNorm1d(n_elements//2))
            seq.add_module('Dropout_2', nn.Dropout(0.25))
            seq.add_module('linear_2', nn.Linear(n_elements//2, stage_2_out))
            self.stage_2.append(seq)
            self.add_module('stage_2_{}'.format(i), seq)

        #dense_in = len(subscripted_views) * stage_2_out
        dense_in = len(subscripted_views) * n_elements


        downsample_factor = 300
        pool_1 = nn.Sequential()
        pool_1.add_module('adaptive_maxpool', nn.AdaptiveMaxPool1d(dense_in//downsample_factor))
        self.pool_1 = pool_1

        linear_1 = nn.Sequential()
        linear_1.add_module('batch_norm_1', nn.BatchNorm1d(dense_in//downsample_factor))
        linear_1.add_module('drop_out_1', nn.Dropout(0.25))
        #linear_1.add_module('linear_1', nn.Linear(dense_in, dense_in//2))
        linear_1.add_module('relu_1', nn.ReLU())
        #linear_1.add_module('batchnorm_2', nn.BatchNorm1d(dense_in//2))
        #linear_1.add_module('drop_out_2', nn.Dropout(0.25))
        #linear_1.add_module('linear_2', nn.Linear(dense_in//2, 200))
        linear_1.add_module('linear_2', nn.Linear(dense_in//downsample_factor, 200))
        linear_1.add_module('softmax_2', nn.Softmax())
        self.linear_1 = linear_1

        #dense_1 = nn.Sequential()
        #dense_1.add_module('Dense', torch.nn.Dense(500), activation='softmax')
        #self.dense_1 = dense_1

#        linear_3 = nn.Sequential()
#        linear_3.add_module('linear', nn.Linear(2000, 500))
#        linear_3.add_module('batchnorm', torch.nn.BatchNorm1d(500))
#        linear_3.add_module('relu', nn.ReLU())
#        linear_3.add_module('drop_out', torch.nn.Dropout(0.5))
#        linear_3.add_module('Softmax', torch.nn.Softmax())
#
#        self.linear_3 = linear_3
#
#        linear_4 = nn.Sequential()
#        linear_4.add_module('linear', nn.Linear(500, 500))
#        linear_4.add_module('batchnorm', torch.nn.BatchNorm1d(500))
#        linear_4.add_module('relu', nn.ReLU())
#        linear_4.add_module('drop_out', torch.nn.Dropout(0.5))
#        linear_4.add_module('Softmax', torch.nn.Softmax())
#        self.linear_4 = linear_4
#
#        linear_5 = nn.Sequential()
#        linear_5.add_module('linear', nn.Linear(500, 500))
#        linear_5.add_module('batchnorm', torch.nn.BatchNorm1d(500))
#        linear_5.add_module('relu', nn.ReLU())
#        linear_5.add_module('drop_out', torch.nn.Dropout(0.5))
#        linear_5.add_module('Softmax', torch.nn.Softmax())
#        self.linear_5 = linear_5
#
#        linear_6 = nn.Sequential()
#        linear_6.add_module('linear', nn.Linear(500, 500))
#        linear_6.add_module('batchnorm', torch.nn.BatchNorm1d(500))
#        linear_6.add_module('relu', nn.ReLU())
#        linear_6.add_module('drop_out', torch.nn.Dropout(0.5))
#        linear_6.add_module('Softmax', torch.nn.Softmax())
#        self.linear_6 = linear_6
#
#
#
#        linear_2 = nn.Sequential()
#        linear_2.add_module('linear', nn.Linear(500, 200))
#        #linear_2.add_module('relu', nn.ReLU())
#        #linear_2.add_module('drop_out', torch.nn.Dropout(0.5))
#        #linear_2.add_module('Softmax', torch.nn.Softmax())
#        #linear_2.add_module('linear', nn.Linear(200, 200))
#
#        self.linear_2 = linear_2

    def forward(self, batch):
        x = [batch[n] for n in self.subscripted_views]
        x = [[self.transform(dgm) for dgm in view_batch] for view_batch in x]

        x = self.pht_sl(x)

        x = torch.cat(x, 1)

        x = [l(xx) for l, xx in zip(self.stage_1, x)]

        x = [l(xx) + xx for l, xx in zip(self.stage_3, x)]

        x = [l(xx) for l, xx in zip(self.stage_4, x)]

        x = [torch.squeeze(torch.max(xx, 1)[0]) for xx in x]

        #x = [l(xx) for l, xx in zip(self.stage_2, x)]
        x = torch.cat(x, 1)
        x = torch.unsqueeze(x, 0)
        x = self.pool_1(x)
        x = torch.squeeze(x)
        x = self.linear_1(x)
        #x = self.dense_1(x)
        #x = self.linear_3(x)
        #x = self.linear_4(x)
        #x = self.linear_5(x)
        #x = self.linear_6(x)
        #x = self.linear_2(x)
        return x


def train_test_from_dataset(dataset,
                            batch_size):
    sample_labels = list(dataset.sample_labels)
    label_encoder = LabelEncoder().fit(sample_labels)
    sample_labels = label_encoder.transform(sample_labels)

    def label_remapper(label):
        return int(label_encoder.transform([label])[0])

    label_map = label_remapper

    collate_fn = PersistenceDiagramProviderCollate(dataset,
                                                   label_map=label_map)

    train_ids = np.array([label_map(image_id)
                          for image_id in dataset.sample_labels
                          if training_data_labels[image_id]])
    test_ids = np.array([label_map(image_id)
                         for image_id in dataset.sample_labels
                         if not training_data_labels[image_id]])

    data_train = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False,
                            sampler=SubsetRandomSampler(train_ids.tolist()))

    data_test = DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collate_fn,
                           shuffle=False,
                           sampler=SubsetRandomSampler(test_ids.tolist()))

    return data_train, data_test


def read_provider(data_path):
    dataset = Provider(dict(), None, dict())
    dataset.read_from_h5(data_path)
    return dataset


def load_data(params):
    rgb = params['colour_mode'] == 'rgb'
    grayscale_wide = params['colour_mode'] == 'grayscale_wide'

    view_name_template = 'dim_0_dir_{}'
    subscripted_views = sorted([view_name_template.format(i)
                               for i in range(params['directions'])])
    assert (str(len(subscripted_views)) in params['data_path'])

    subscripted_views_colour = []
    if rgb or grayscale_wide:
        for colour in colours:
            for view in subscripted_views:
                subscripted_views_colour.append(view + '_{}'.format(colour))
    else:
        subscripted_views_colour = subscripted_views

    print('Loading providers')
    data_paths = []
    datasets = []
    if rgb or grayscale_wide:
        if grayscale_wide:
            print("Accuracy: Modified grayscale")
        dataset_files = []
        for colour in colours:
            if rgb:
                dataset_files.append(os.path.join(params['data_path'],
                                                  colour + '.h5'))
            else:
                dataset_files.append(os.path.join(params['data_path'],
                                                  'gray' + '.h5'))
        with multiprocessing.Pool() as pool:
            datasets = pool.map(read_provider, dataset_files)

        print("Merging providers")
        merged_dataset = Provider(dict(), None, dict())
        for i, colour in enumerate(colours):
            for view in datasets[i].data_views:
                key = view + "_{}".format(colour)
                merged_dataset.add_view(key, datasets[i].data_views[view])

    else:
        merged_dataset = read_provider(os.path.join(params['data_path'],
                                                    'gray.h5'))

    print('Create data loader...')
    data_train, data_test = \
        train_test_from_dataset(merged_dataset,
                                batch_size=params['batch_size'])

    return data_train, data_test, subscripted_views_colour


def setup_trainer(model, params, data_train, data_test):
    optimizer = optim.SGD(model.parameters(),
                          lr=params['lr_start'],
                          momentum=params['momentum'])

    loss = nn.CrossEntropyLoss()

    trainer = tr.Trainer(model=model,
                         optimizer=optimizer,
                         loss=loss,
                         train_data=data_train,
                         n_epochs=params['epochs'],
                         cuda=params['cuda'],
                         variable_created_by_model=True,
                         sat_key=params['sat_key'])

    def determine_lr(self, **kwargs):
        epoch = kwargs['epoch_count']
        if epoch % params['lr_ep_step'] == 0:
            return params['lr_start'] / 2 ** (epoch / params['lr_ep_step'])

    lr_scheduler = LearningRateScheduler(determine_lr, verbose=True)
    lr_scheduler.register(trainer)

    progress = ConsoleBatchProgress()
    progress.register(trainer)

    prediction_monitor_test = PredictionMonitor(data_test,
                                                verbose=True,
                                                eval_every_n_epochs=1,
                                                variable_created_by_model=True)
    prediction_monitor_test.register(trainer)
    trainer.prediction_monitor = prediction_monitor_test

    return trainer


def train_network(parameters):
    if torch.cuda.is_available():
        parameters['cuda'] = True
    #else:
        # Let the CPU cluster chew large pieces
    #    parameters['batch_size'] = 128

    print(params)

    print('Data setup...')
    data_train, data_test, subscripted_views = load_data(parameters)

    print("Creating network")
    model = SLayerRgbNN(subscripted_views, parameters['directions'])

    print("Creating trainer")
    trainer = setup_trainer(model, params, data_train, data_test)

    print("Running training")

    trainer.run()

    print("Saving model")
    model_path = os.path.join(parameters['data_path'],
                              "model" + serialise_params(parameters) +
                              ".torch")
    torch.save(model.state_dict(), model_path)

    last_10_accuracies = \
        list(trainer.prediction_monitor.accuracies.values())[-10:]
    mean = np.mean(last_10_accuracies)

    return mean


if __name__ == '__main__':
    params = _parameters()
    histogram_normalised = True
    outpath = os.path.join(os.path.dirname(__file__), 'h5images_short')
    params['data_path'] = get_folder_string(32, params['resampled_size'],
                                            outpath,
                                            histogram_normalised)
    if params['colour_mode'] == 'rgb':
        if not os.path.exists(os.path.join(params['data_path'], 'red.h5')):
            rotate_all_persistence_diagrams(params['directions'], params['resampled_size'],
                                            outpath, histogram_normalised, rgb=True)
    elif params['colour_mode'] == 'grayscale' or \
            params['colour_mode'] == 'grayscale_wide':
        if not os.path.exists(os.path.join(params['data_path'], 'gray.h5')):
            rotate_all_persistence_diagrams(params['directions'],
                                            params['resampled_size'], outpath,
                                            histogram_normalised, rgb=False)
    else:
        raise RuntimeError('Parameter colour_mode = {} not recognised!'
                           .format(params['colour_mode']))
    mean = train_network(params)
    print("Mean is {}".format(mean))
