import torch

w = torch.tensor(1.0, requires_grad = True)
b = torch.tensor(0.0, requires_grad = True)

x = torch.tensor(2.0)
y = torch.tensor(6.0)

prediction = w*x + b

loss = 0.5 *(prediction - y) ** 2

print("Prediction: ", prediction.item())
print("Loss: ", loss.item())


loss.backward()

print("Gradient of w: ", w.grad.item())
print("Gradient of b: ", b.grad.item())



# SGD Update


learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad


print("updated w: ", w.item())
print("updated b: ", b.item())

new_prediction = w*x + b

loss = 0.5 * (new_prediction - y) ** 2

print("new prediction: ", new_prediction.item())
print("loss: ", loss.item())


