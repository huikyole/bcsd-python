import numpy as np

class QMap():
    def __init__(self, step=0.5):
        self.step = step

    def fit(self, x, y, z, axis=None):
        """ Calculate percentiles from observed and modeled datasets
        and their differences

        :param x: 3d observed dataset
        :type x: :py:class:`~xarray.DataArray` 
        :param y: 3d modeled dataset for the present climate
        :type y: :py:class:`~xarray.DataArray` 
        :param z: 3d modeled dataset for the future climate
        :type z: :py:class:`~xarray.DataArray` 
        """ 
        if axis not in (None, 0):
            raise ValueError("Axis should be None or 0")
        self.axis = axis
        steps = np.arange(self.step, 100, self.step)
        self.x_map = np.nanpercentile(x, steps, axis=axis)
        self.y_map = np.nanpercentile(y, steps, axis=axis)
        self.z_map = np.nanpercentile(y, steps, axis=axis)
        #self.y_bias_map = self.y_map - self.x_map # (biases in percentiles)
        self.x_to_y_ratio_map = np.zeros(self.x_map.shape)
        self.x_to_y_ratio_map[self.y_map!=0] = self.x_map[self.y_map!=0]/self.y_map[self.y_map!=0]
        self.x_to_y_ratio_map[self.x_map<-900.] = np.nan
        return self

    def predict(self, z):
        idx = [np.abs(val - self.z_map).argmin(axis=self.axis) for val in z]
        if self.axis == 0:
            nt, nx = z.shape
            out = np.zeros(z.shape)
            for it in np.arange(nt):
                for ix in np.arange(nx):
                    out[it,ix] = self.x_to_y_ratio_map[idx[it][ix],ix]*z[it,ix] 
        else:
            out = self.x_to_y_ratio_map[idx]*z
        return out

def test_qmap():
    np.random.seed(0)
    x = np.random.normal(10, size=(10,20))
    y = np.random.normal(100, size=(10, 20))
    z = np.random.normal(100, size=(10, 20))
    mapped = np.zeros(x.shape)
    for j in range(x.shape[1]):
        qmap = QMap()
        qmap.fit(x[:,j], y[:,j], z[:,j])
        mapped[:, j] = qmap.predict(z[:,j])

if __name__ == "__main__":
    test_qmap()
