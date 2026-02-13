from hilbert import HilbertScan
import matplotlib.pyplot as plt
import numpy as np

def plot_hilbert_curve(n=8, save_path=None):
    hilbert = HilbertScan()
    coords = [hilbert._d2xy(n, d) for d in range(n*n)]
    coords = np.array(coords)

    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(6,6))
    plt.plot(x, y, '-o', color='blue', markersize=8)

    for i, (xi, yi) in enumerate(coords):
        plt.text(xi, yi, str(i), fontsize=8, ha='center', va='center', color='red')

    plt.title(f'Hilbert Curve of order {int(np.log2(n))} ({n}x{n} grid)')
    plt.gca().invert_yaxis()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Hilbert curve saved as {save_path}")
    plt.show()

def average_consecutive_distance(n=64, hilbert=True):
    if hilbert:
        hilbert_curve = HilbertScan()
        coords = np.array([hilbert_curve._d2xy(n, d) for d in range(n*n)])
    else:
        coords = np.array([(i % n, i // n) for i in range(n*n)])

    dists = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    avg_dist = np.mean(dists)
    return dists, avg_dist

if __name__ == "__main__":
    
    plot_hilbert_curve(8, save_path="Hilbert8x8.png")

    row_dists, row_avg = average_consecutive_distance(64, hilbert=False)
    hilbert_dists, hilbert_avg = average_consecutive_distance(64, hilbert=True)

    print("\nAverage distance between consecutive pixels (64x64 grid):")
    print(f"Row-major scanning: {row_avg:.2f}")
    print(f"Hilbert scanning : {hilbert_avg:.2f}")

    print("\nObservation: Hilbert scanning keeps pixels that are close in 2D also close in the 1D sequence. This should help sequence models like S4 pick up local patterns easier.")
