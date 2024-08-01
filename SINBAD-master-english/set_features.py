import torch
import torch.nn.functional as F

'''
This custom PyTorch module, CumulativeSetFeatures, is used to convert time series data into cumulative distribution functions (CDF) and points based
A feature set of bits.

1. Initialization:
-n_channels: Number of channels to enter the time series data.
- n_projections: Number of random projections, default is 100.
-n_quantiles: indicates the number of quantiles. The default value is 20.
-is_projection: A Boolean value indicating whether a random projection should be used to transform data.
- projections: If is_projection is True, the projection matrix used to transform the data will be generated randomly.
2.fit method:
- This method is used to fit the parameters of the model according to the data provided, mainly to calculate the minimum and maximum values of the features for later use in the calculation of quantile thresholds.
- If is_projection is True, a random projection is used to transform the input data into a new feature representation.
3.forward Method:
- During forward propagation, the model transforms the input data according to the parameters calculated during training.
- If is_projection is True, the projection learned during training is used.
For each quantile, a threshold is calculated and based on the threshold, the input data is converted into a binary feature set indicating whether the data is less than that threshold.
Calculate the cumulative distribution function (CDF) of the feature set under each quantile.
- Returns the CDF and feature set.
This module can be used for the pre-processing of time series data, so that subsequent machine learning models can better understand and utilize the dynamic characteristics of time series.
In the code, the convolution operation of PyTorch is used to implement random projection, and the quantization function of PyTorch is used to calculate quantile threshold.
'''
class CumulativeSetFeatures(torch.nn.Module):
    def __init__(self, n_channels, n_projections=100, n_quantiles=20, is_projection=True):
        self.n_channels = n_channels
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        self.projections = torch.randn(self.n_projections,  self.n_channels, 1)
        self.is_projection = is_projection

    def fit(self, X):
        # Calculate the feature representation after projection or transposition
        if self.is_projection:
            a = F.conv1d(X, self.projections).permute((0, 2, 1))
            a = a.reshape((-1, self.n_projections))
        else:
            a = X.permute((0, 2, 1))
            a = a.reshape((a.shape[0]*a.shape[1], -1))
        # Calculate quantile thresholds for features
        self.min_vals = torch.quantile(a, 0.01, dim=0)
        self.max_vals = torch.quantile(a, 0.99, dim=0)


    def forward(self, X):
        # Calculate a projection or use a feature representation of the original input
        if self.is_projection:
            a = F.conv1d(X, self.projections)
        else:
            a = X
        # Initializes the cumulative distribution function (CDF) and the feature set
        cdf = torch.zeros((a.shape[0], a.shape[1], self.n_quantiles))
        set = torch.zeros((a.shape[0], a.shape[1], X.shape[-1], self.n_quantiles,))
        # The threshold for each quantile is calculated and the feature set and CDF are constructed
        for q in range(self.n_quantiles):
            threshold = self.min_vals + (self.max_vals - self.min_vals) * (q + 1) / (self.n_quantiles + 1)
            set[:, :, :, q] = (a < threshold.unsqueeze(0).unsqueeze(2)).float()
            cdf[:, :, q] = set[:, :, :, q].mean(2)
        # Rearrange and reshape the dimension of the feature set and CDF
        set = torch.transpose(set, 2, 1)
        set = set.reshape((X.shape[0], X.shape[-1], -1))
        set = torch.transpose(set, 2, 1).numpy()
        cdf = cdf.reshape((X.shape[0], -1)).numpy()

        return cdf, set


