import numpy as np
from numpy.polynomial.legendre import leggauss


def collocation1d(n:int, r:int) -> tuple[np.array, np.array]:
    ''' Get One Variable Partition and Collocation Points on [0, 1]

    Args:
        - n: <int> number of partition.
        - r: <int> dimension of polynomials. 
    Returns:
        - partition: <np.array> [n + 1] location of partition points. 
        - collocation: <np.array> [n * (r - 1)] location of collocation points.
    '''
    # Partition Points
    partition = np.linspace(0., 1., num=n + 1, endpoint=True)
    # Collocation Points
    gaussian = (np.array(leggauss(r - 1)[0]) + 1)/2
    collocation = []
    for i in range(n):
        for j in range(r - 1):
            collocation.append(partition[i] + 1/n * gaussian[j])
    collocation = np.squeeze(np.array(collocation))
    return partition, collocation


def collocation2d(nx:int, ny:int, r:int) -> tuple[np.array, np.array]:
    ''' Get Two Variable Partition and Collocation Points on [0, 1] * [0, 1]

    Args:
        - nx: <int> number of partition on variable x.
        - ny: <int> number of partition on variable y.
        - r: <int> dimension of polynomials.
    Returns:
        - partition: <np.array> [nx+1, ny+1, 2] location of partition points.
        - collocation: <np.array> [nx, ny, r-1, r-1, 2] location of collocation points.
    '''
    # Partition Points
    partition = []
    for i in range(nx + 1):
        partition_line = []
        for j in range(ny + 1):
            partition_line.append([i * (1/nx), j * (1/ny)])
        partition.append(partition_line)
    partition = np.array(partition)

    # Collocation Points
    gaussian = (np.array(leggauss(r-1)[0]) + 1)/2
    collocation = []
    for i in range(nx):
        cell_x = []
        for j in range(ny):
            cell_y = []
            for p in range(r-1):
                cell_line = []
                for q in range(r-1):
                    x = partition[i, j, 0] + gaussian[p] * (1 / nx)
                    y = partition[i, j, 1] + gaussian[q] * (1 / ny)
                    cell_line.append([x, y])
                cell_y.append(cell_line)
            cell_x.append(cell_y)
        collocation.append(cell_x)
    collocation = np.array(collocation)
    return partition, collocation


if __name__ == '__main__':
    partition, collocation = collocation1d(3, 3)
    print(f'1D Collocation Partition Size {partition.shape}')
    print(f'1D Collocation Collocation Size {collocation.shape}')

    partition, collocation = collocation2d(4, 4, 3)
    print(f'2D Collocation Partition Size {partition.shape}')
    print(f'2D Collocation Collocation Size {collocation.shape}')
