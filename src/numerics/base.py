import torch


class bd0:
    """Base Function in Left Boundary."""

    def __init__(self, x1, x2, device):
        self.x1 = torch.tensor(x1, device=device)
        self.x2 = torch.tensor(x2, device=device)
        self.dx = torch.tensor(x2 - x1, device=device)

    def f(self, x):
        if x >= self.x1 and x <= self.x2:
            return (x - self.x1) * (self.x2 - x) ** 2 / self.dx ** 2
        else:
            return 0


class bd1:
    """Base Function in Right Boundary."""

    def __init__(self, x1, x2, device):
        self.x1 = torch.tensor(x1, device=device)
        self.x2 = torch.tensor(x2, device=device)
        self.dx = torch.tensor(x2 - x1, device=device)

    def f(self, x):
        if x >= self.x1 and x <= self.x2:
            return (x - self.x2) * (x - self.x1) ** 2 / self.dx ** 2
        else:
            return 0


class ce0:
    """Base Function in Center. Deveriate: 0."""

    def __init__(self, x0, x1, x2, device):
        self.x0 = torch.tensor(x0, device=device)
        self.x1 = torch.tensor(x1, device=device)
        self.x2 = torch.tensor(x2, device=device)
        self.dx1 = torch.tensor(x1 - x0, device=device)
        self.dx2 = torch.tensor(x2 - x1, device=device)

    def f(self, x):
        if x >= self.x0 and x <= self.x1:
            return (self.dx1 + 2 * (self.x1 - x)) * (x - self.x0) ** 2 / self.dx1 ** 3
        elif x >= self.x1 and x <= self.x2:
            return (self.dx2 + 2 * (x - self.x1)) * (self.x2 - x) ** 2 / self.dx2 ** 3
        else:
            return 0


class ce1:
    """Base Function in Center. Deveriate: 1."""

    def __init__(self, x0, x1, x2, device):
        self.x0 = torch.tensor(x0, device=device)
        self.x1 = torch.tensor(x1, device=device)
        self.x2 = torch.tensor(x2, device=device)
        self.dx1 = torch.tensor(x1 - x0, device=device)
        self.dx2 = torch.tensor(x2 - x1, device=device)

    def f(self, x):
        if x >= self.x0 and x <= self.x1:
            return (x - self.x1) * (x - self.x0) ** 2 / self.dx1 ** 2
        elif x >= self.x1 and x <= self.x2:
            return (x - self.x1) * (self.x2 - x) ** 2 / self.dx2 ** 2
        else:
            return 0
