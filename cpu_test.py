import multiprocessing as mp
import time
import math
import os

# Heavy CPU-bound task
def cpu_task(worker_id, iterations):
    result = 0.0
    for i in range(iterations):
        result += math.sqrt(i % 1000) * math.sin(i)
    return result

def main():
    NUM_CORES = 32          # set to your CPU core count
    ITERATIONS = 50_000_000_000 # increase/decrease for longer/shorter test

    print(f"PID: {os.getpid()}")
    print(f"Using {NUM_CORES} CPU cores")
    print(f"Iterations per worker: {ITERATIONS:,}")

    start = time.time()

    with mp.Pool(processes=NUM_CORES) as pool:
        results = pool.starmap(
            cpu_task,
            [(i, ITERATIONS) for i in range(NUM_CORES)]
        )

    elapsed = time.time() - start

    print("\nBenchmark complete")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Result checksum: {sum(results):.4f}")

if __name__ == "__main__":
    main()
