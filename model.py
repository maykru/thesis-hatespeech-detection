import torch
import torch.nn as nn
import math


class CNN(nn.ModuleList):
    def __init__(self, weights_matrix, hidden_size, number_feature):
        super(CNN, self).__init__()
        # intialize embedding layer with required size
        self.vocab_size = weights_matrix.shape[0]   # no. words in dataset
        self.vector_size = weights_matrix.shape[1]  # dimension of vectors

        self.embedding_layer = nn.Embedding(self.vocab_size + 1, self.vector_size)
        # initialise embedding layer using pre-trained weights
        self.embedding_layer.weight = nn.Parameter(torch.tensor(weights_matrix, dtype=torch.float32))
        # disable learning bc pre-trained
        # self.embedding_layer.weight.requires_grad=False

        self.stride = 2
        self.dropout = nn.Dropout(0.5)
        self.seq_len = 13
        self.out_size = 32

        # kernels
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # convolution layers
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

        # pooling layers
        self.pooling_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pooling_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pooling_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pooling_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        # feature layer
        self.feature_layer = nn.Linear(number_feature, number_feature).float()

        # combination layer
        self.combined_layer = nn.Linear(self.in_features_fc() + number_feature, hidden_size).float()

        # fully connected layer
        # self.fc = nn.Linear(self.in_features_fc(), 1)              #without features
        self.fc = nn.Linear(hidden_size, 1)  # with features

    def in_features_fc(self):
        self.embedding_size = 200
        # Calculate size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calculate size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calculate size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calculate size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_size

    def forward(self, embedding_input, feature_input):
        x = self.embedding_layer(embedding_input)

        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pooling_1(x1)

        x2 = self.conv_2(x)
        x2 = torch.relu(x2)
        x2 = self.pooling_2(x2)

        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pooling_3(x3)

        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pooling_4(x4)

        feature_layer = self.feature_layer(feature_input)
        result = torch.cat((x1, x2, x3, x4), 2)
        result = result.reshape(result.size(0), -1)
        combined = torch.cat((result, feature_layer), 1)
        combined_layer = self.combined_layer(combined)
        out = self.fc(combined_layer)
        # out = self.fc(result)  	            #without number features
        out = self.dropout(out)
        out = torch.sigmoid(out)
        return out.squeeze()