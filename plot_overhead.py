import subprocess
import matplotlib.pyplot as plt

functions = {
    "sin(x)": "SIN",
    "exp(x*x)": "EXP2",
    "ln(x+1) + tanh(x)": "LOGTANH",
    "sigmoid(exp(x))": "SIGEXP",
    "x*sin(x) + x*exp(x)": "XCOMBO"
}

sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
results = {name: {"plain": [], "dual": []} for name in functions}

for name, macro in functions.items():
    print(f"Benchmarking {name}...")
    for n in sizes:
        # Plain version
        subprocess.run(["g++", "../dual_test.cpp", "-o", "dual_plain", f"-D{macro}", "-DPLAIN_ONLY", "-O2"], check=True)
        plain_output = subprocess.check_output(["./dual_plain", str(n)]).decode().strip()
        results[name]["plain"].append(float(plain_output))

        # Dual version
        subprocess.run(["g++", "../dual_test.cpp", "-o", "dual_dual", f"-D{macro}", "-O2"], check=True)
        dual_output = subprocess.check_output(["./dual_dual", str(n)]).decode().strip()
        results[name]["dual"].append(float(dual_output))

# Plot
for name in functions:
    plt.plot(sizes, results[name]["plain"], label=f"{name} (no AD)", linestyle="--", marker="o")
    plt.plot(sizes, results[name]["dual"], label=f"{name} (dual)", linestyle="-", marker="x")

plt.xlabel("Input size (n)")
plt.ylabel("Time (ms)")
plt.title("Overhead of Dual Numbers vs Plain Function Evaluation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dual_number_overhead.png")
plt.show()