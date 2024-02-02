import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi

x = torch.linspace(-pi, pi, 100)
y = torch.sin(x)

w = torch.rand(4, requires_grad=True)
pw = torch.tensor([0, 1, 2, 3])
learning_rate = 1e-6

ps = []

with torch.no_grad():
	p = (x.unsqueeze(-1).pow(pw) * w).sum(1)
	loss = (p - y).pow(2).sum()

for i in range(600):
	p = (x.unsqueeze(-1).pow(pw) * w).sum(1)
	loss = (p - y).pow(2).sum()
	if i % 10 == 0: ps.append(p)
	loss.backward()
	with torch.no_grad():
		for i in range(4):
			w[i] -= learning_rate * w.grad[i]
		w.grad = None

while loss.item() > 3:
	for i in range(600):
		p = (x.unsqueeze(-1).pow(pw) * w).sum(1)
		loss = (p - y).pow(2).sum()
		if i % 10 == 0: ps.append(p)
		loss.backward()
		with torch.no_grad():
			for i in range(4):
				w[i] -= learning_rate * w.grad[i]
			w.grad = None

fig, ax = plt.subplots()
ax.set_ylim(-2, 2)
ax.plot(x, y)
line_pred = ax.plot(x, ps[0].data)

newps = []

s = 0
e = 60

loop = 1

while True:
	if e > len(ps): e = len(ps)
	tmp = ps[s:e]
	interval = (e-s) / 60
	for i in range(60):
		newps.append(tmp[int(i*interval)])
	if e >= len(ps):
		break
	else:
		s = e - 1
		e = 60 * 2**loop
		loop += 1


def update(frame):
    line_pred[0].set_ydata(newps[frame].data)

anim = animation.FuncAnimation(fig=fig, func=update, frames=len(newps), interval=1000/60, repeat=True)

plt.show()
