import torch
import numpy as np
import sys

sys.path.append(".")
from itertools import product
from src.numerics.collocation import collocation1d, collocation2d
from src.numerics.base import bd0, bd1, ce0, ce1


class osc1d(object):
    """
    Orthogonal Spline Collocation 1D
    Args:
        - p <torch.tensor>: a 1-D array of real value as the partition position.
        - c <torch.tensor>: a 1-D array of real value as the collocation position.
        - y <torch.tensor>: a 1-D array of real value as the value at the collocation
            position. The length of y should be equal to the length of x.
        - b1 <torch.float> left boundary value.
        - b2 <torch.float> left boundary value.
    """

    def __init__(self, p, c, y, b1, b2, device):
        # TODO: input check
        self.p = p
        self.n = len(self.p) - 1
        self.r = int(len(c) / self.n) + 1
        self.device = device
        weight, value = self._generator1d(self.p, c, y, self.n, self.r, b1, b2)
        weight = weight.to(device, torch.float32)
        self.para = torch.linalg.solve(weight, value)

    def __call__(self, x):
        x = x.to(self.device)
        result = torch.empty(0).to(self.device)
        for i in range(self.n):
            if i < self.n - 1:
                x_part = x[torch.logical_and(x >= self.p[i], x < self.p[i + 1])]
            else:
                x_part = x[torch.logical_and(x >= self.p[i], x <= self.p[i + 1])]
            y = torch.zeros(len(x_part)).to(self.device)
            for j in range(self.r + 1):
                y += self.para[i * (self.r + 1) + j] * x_part ** j
            result = torch.cat((result, y))
        return result

    def _generator1d(self, p, c, y, n, r, b1, b2):
        """Generator the Algebraic Equation Matrix"""
        weight = []
        value = []

        # Partition Points
        for i in range(n):
            start_idx = i * (r - 1)
            end_idx = (i + 1) * (r - 1)

            # Collocation Points
            for j, xi in enumerate(c[start_idx:end_idx]):
                coll_idx = i * (r - 1) + j
                weight.append(self._substitute(xi, coll_idx, n, r, 0))
                value.append(y[coll_idx])

            # C1 Continuous
            if i == n - 1:
                continue
            c1_value, c1_dev = self._continuous(p[i + 1], i + 1, n, r)
            weight.append(c1_value)
            weight.append(c1_dev)
            value.append(torch.tensor(0))
            value.append(torch.tensor(0))

        # Boundary Condition
        left_bd, right_bd = self._boundary(n, r)
        weight.insert(0, left_bd)
        weight.append(right_bd)
        value.insert(0, torch.tensor(b1))
        value.append(torch.tensor(b2))
        return torch.stack(weight), torch.stack(value)

    def _substitute(self, x: float, k: int, n: int, r: int, ord: int):
        """Generator the Equation of Substituting Collocation Value into the Original Equation

        Args:
            - x: <float> value of the collocation point gonna to use.
            - k: <int> index of the collocation point gonna to use.
            - n: <int> number of partition.
            - r: <int> order of polynomial.
            - ord: <int> order of deverate, should be non-negetive.
        """
        result = torch.zeros(n * (r + 1))
        poly_idx = int(k / (r - 1))
        for i in range(r + 1):
            start_idx = poly_idx * (r + 1)
            temp = x ** (i - ord)
            for j in range(ord):
                temp *= i - j
            result[start_idx + i] = temp
        return result

    def _continuous(
        self, x: float, k: int, n: int, r: int
    ) -> tuple[torch.tensor, torch.tensor]:
        """Generate the Equation of C1 Continuouse

        Args:
            - x: <float> value of the partition point gonna to use.
            - k: <int> index of the partition point gonna to use.
            - n: <int> number of partition.
            - r: <int> order of polynomial.
        Returns:
            - value: <np.array> [n * (r + 1)] value equal equiation
            - dev: <np.array> [n * (r + 1)] deverate equal equition
        """
        value = torch.zeros(n * (r + 1))
        dev = torch.zeros(n * (r + 1))

        # Previous Polynomial
        poly_idx = (k - 1) * (r + 1)
        for i in range(r + 1):
            value[poly_idx + i] = x ** i
            dev[poly_idx + i] = i * x ** (i - 1)

        # Next Polynomial
        poly_idx = k * (r + 1)
        for i in range(r + 1):
            value[poly_idx + i] = -(x ** i)
            dev[poly_idx + i] = -i * x ** (i - 1)
        return value, dev

    def _boundary(self, n: int, r: int):
        """Generate the Equation of Boundary Condition

        Args:
            - n: <int> number of partition.
            - r: <int> order of polynomial.
        Returns:
            - left: <np.array> [n * (r + 1)] left boundary condition equation.
            - right: <np.array> [n * (r + 1)] right boundary condition equation.
        """
        left = torch.zeros(n * (r + 1))
        right = torch.zeros(n * (r + 1))

        # Left Boundary Condition
        idx = 0
        for i in range(r + 1):
            left[i] = 0 ** i

        # Right Boundary Condition
        idx = (n - 1) * (r + 1)
        for i in range(r + 1):
            right[idx + i] = 1 ** i
        return left, right


class osc2d(object):
    """
    Orthogonal Spline Collocation 2D
    Args:
        - p <torch.tensor>: a 1-D array of real value as the partition position.
        - c <torch.tensor>: a 1-D array of real value as the collocation position.
        - y <torch.tensor>: a 1-D array of real value as the value at the collocation
            position. The length of y should be equal to the length of x.
        - b1 <torch.float> left boundary value.
        - b2 <torch.float> left boundary value.
    """

    def __init__(self, p, c, z, device):
        self.device = device
        self.r = c.size(2) + 1
        self.nx = p.size(0) - 1
        self.ny = p.size(1) - 1
        weight, value = self._generator2d(p, c, z, self.nx, self.ny, self.r)
        self.value = value.to(device, torch.float32)
        self.weight = weight.to(device, torch.float32)
        self.para = (
            torch.linalg.solve(self.weight, self.value)
            .reshape(((self.nx - 1) * (self.r - 1) + 2, -1))
            .to(device)
        )

    def __call__(self, base):
        """Generate Simulation Results on 2D Range."""
        result = torch.einsum("ij,ijmn->mn", self.para, base)
        return result

    def _generator2d(self, p, c, z, nx, ny, r):
        """Generator the Algebraic Equation Matrix
        Args:
            - p: <torch.tensor> [nx, ny, 2] partition points.
            - c: <torch.tensor> [nx, ny, rx-1, ry-1, 2] collocation points.
            - z: <torch.tensor> [nx, ny, rx-1, ry-1] value at the collocation points.
            - nx: <int> number of partition in x direction.
            - ny: <int> number of partition in y direction.
            - r: <int> order of polynomial.
        """
        self.x_base = self._base(nx)
        self.y_base = self._base(ny)
        weight_list = []
        value_list = []
        # For each collocation point
        for x_part, y_part, x_coll, y_coll in product(
            range(nx), range(ny), range(r - 1), range(r - 1)
        ):
            weight = torch.zeros((nx * (r - 1), ny * (r - 1))).to(self.device)
            coll = c[x_part, y_part, x_coll, y_coll]
            coll_x = coll[0]
            coll_y = coll[1]
            for x_base_idx, y_base_idx in product(
                range(nx * (r - 1)), range(ny * (r - 1))
            ):
                weight[x_base_idx, y_base_idx] += self.x_base[x_base_idx].f(
                    coll_x
                ) * self.y_base[y_base_idx].f(coll_y)
            weight_list.append(weight.flatten())
            value_list.append(z[x_part, y_part, x_coll, y_coll])
        weight_list = torch.stack(weight_list, dim=0)
        value_list = torch.stack(value_list, dim=0)
        return weight_list, value_list

    def _base(self, n: int) -> list:
        """Generate the Base Function."""
        x = np.linspace(0, 1, n + 1, endpoint=True)
        base_list = []
        base_list.append(bd0(x[0], x[1], self.device))
        for i in range(1, n):
            base_list.append(ce0(x[i - 1], x[i], x[i + 1], self.device))
            base_list.append(ce1(x[i - 1], x[i], x[i + 1], self.device))
        base_list.append(bd1(x[-2], x[-1], self.device))
        return base_list


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    case = "2d"

    if case == "1d":
        n = 3
        r = 3
        b1 = 0
        b2 = 0
        f = lambda x: 10 * x ** 3 + 14 * x ** 2 + 34 * x - 26
        u = lambda x: 10 * x ** 3 - 16 * x ** 2 + 6 * x

        # Generate partition and collocation
        p, c = collocation1d(n, r)
        y = torch.tensor(u(c), dtype=torch.float32)
        y.requires_grad_()

        # Collocation
        f_ = osc1d(p, c, y, b1, b2)

        # Get prediction value
        x = torch.linspace(0, 1, 256)
        gt = u(x)
        pred = f_(x)

        # Print loss
        loss = torch.sum((gt - pred) ** 2)
        loss.backward()
        print(loss)
        print(y.grad)

    elif case == "2d":
        nx = 4
        ny = 4
        dx = 1 / nx
        dy = 1 / ny
        sim_x = 256
        sim_y = 256
        r = 3
        u = lambda x, y: 10 * (x ** 2 * y ** 2 - x ** 2 * y - x * y ** 2 + x * y)

        # Prepare for the base
        start = time.process_time()
        x = torch.linspace(0, 1, sim_x)
        split_x = int(sim_x / nx)
        sx = []
        for i in range(nx + 1):
            sx.append(torch.ones(split_x) * i / nx)

        px_l0 = torch.cat(sx[:nx])
        px_l1 = torch.cat(sx[1:])
        px_l2 = torch.cat(sx[2:] + sx[-1:])

        px_r0 = torch.cat(sx[:1] + sx[: nx - 1])
        px_r1 = px_l0
        px_r2 = px_l1

        base_xl0 = (dx + 2 * (px_l1 - x)) * (x - px_l0) ** 2 / dx ** 3
        base_xr0 = (dx + 2 * (x - px_r1)) * (px_r2 - x) ** 2 / dx ** 3
        base_xl1 = (x - px_l1) * (x - px_l0) ** 2 / dx ** 2
        base_xr1 = (x - px_r1) * (px_r2 - x) ** 2 / dx ** 2

        base_start = torch.cat(
            (
                x[:split_x] * (sx[1] - x[:split_x]) ** 2 / dx ** 2,
                torch.zeros(sim_x - split_x),
            )
        )
        base_end = torch.cat(
            (
                torch.zeros(sim_x - split_x),
                (x[-split_x:] - sx[-1]) * (x[-split_x:] - sx[-2]) ** 2 / dx ** 2,
            )
        )
        base_list = [base_start]
        for i in range(nx - 1):
            base0 = torch.cat(
                (
                    torch.zeros(i * split_x),
                    base_xl0[i * split_x : (i + 1) * split_x],
                    base_xr0[i * split_x : (i + 1) * split_x],
                    torch.zeros((nx - 2 - i) * split_x),
                )
            )
            base1 = torch.cat(
                (
                    torch.zeros(i * split_x),
                    base_xl1[i * split_x : (i + 1) * split_x],
                    base_xr1[i * split_x : (i + 1) * split_x],
                    torch.zeros((nx - 2 - i) * split_x),
                )
            )
            base_list.append(base0)
            base_list.append(base1)
        base_list.append(base_end)
        base_x = torch.stack(base_list, dim=0)

        # If y has the same partition as x
        base_y = base_x
        base = torch.einsum("ij,kl->ikjl", base_x, base_y).to("cuda:0")
        checkpoint = time.process_time()
        print(f"Calculate Base Time: {checkpoint - start}")

        # Generate partition and collocation
        p, c = collocation2d(nx, ny, r)
        p = torch.tensor(p)
        c = torch.tensor(c)
        z = torch.zeros(c.size()[:-1], dtype=torch.float32)
        for x_part, y_part, x_coll, y_coll in product(
            range(nx), range(ny), range(r - 1), range(r - 1)
        ):
            z[x_part, y_part, x_coll, y_coll] = u(
                c[x_part, y_part, x_coll, y_coll, 0],
                c[x_part, y_part, x_coll, y_coll, 1],
            )
        z.requires_grad_()

        # Collocation
        checkpoint = time.process_time()
        f_ = osc2d(p, c, z, "cuda:0")
        print(f"Sovling Time: {time.process_time() - checkpoint}")

        # Get prediction value
        gt = u(torch.linspace(0, 1, sim_x), torch.linspace(0, 1, sim_y)).to("cuda:0")
        checkpoint = time.process_time()
        pred = f_(base)
        print(f"Simulate Time: {time.process_time() - checkpoint}")

        # Print loss
        loss = torch.sum((gt - pred) ** 2)
        checkpoint = time.process_time()
        loss.backward()
        print(f"Backward Time: {time.process_time() - checkpoint}")
        print(loss)
        print(z.grad)
        # plt.matshow(gt.to('cpu').detach().numpy())
        # plt.savefig('gt.png')
        plt.matshow(pred.to("cpu").detach().numpy())
        plt.savefig("pred.png")
