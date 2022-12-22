import pytorch_forecasting.metrics as m

a = m.SMAPE()

print(a.loss(1,1))

