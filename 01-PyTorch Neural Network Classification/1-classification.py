from sklearn.datasets import make_circles
import pandas as pd

n_samples = 1000

X, y = make_circles(n_samples, noise=.03, random_state=42)

print(X[:5])
print(y[:5])

circles = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "y": y
})

print(circles.head(10))
print(circles.y.value_counts())