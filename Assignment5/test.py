import deeplake

ds = deeplake.load('hub://activeloop/not-mnist-large')

print(ds)
print(ds.images.shape, ds.labels.shape)