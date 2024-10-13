import torch
import torch.optim as optim

## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 6, 10, 12, 15])
Y = torch.tensor([3, 5, 7, 11, 12, 13])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.01)
iterations = 500

for i in range(iterations):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    # 2. gradijent neovisan o broju primjera ako ga reciprocno skaliramo brojem primjera
    loss = torch.mean(diff**2)

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    # ---- 4. ručno računanje gradijenata ----
    a_grad = torch.mean(2*(Y - a*X - b)*(-X))
    b_grad = torch.mean(2*(Y - a*X - b)*(-1))
    # -------------------------------------

    print(f'----------------step: {i}--------------------')
    print(f'loss: {loss}')
    print(f'Y_: {Y_.detach().numpy()}')
    print(f'a: {a.detach().numpy()}')
    print(f'b: {b.detach().numpy()}')
    # 3. ispis gradijenata
    print(f'a.grad: {a.grad.numpy()}')
    print(f'b.grad: {b.grad.numpy()}')
    # 4. ispis ručno izračunatih gradijenata
    print(f'a_grad: {a_grad}')
    print(f'b_grad: {b_grad}\n')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

