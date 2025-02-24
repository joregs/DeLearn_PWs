import numpy as np
import matplotlib.pyplot as plt

# a : compute derivate and sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Générer des valeurs de z
z_values = np.linspace(-10, 10, 100)
sigma_values = sigmoid(z_values)
deriv_values = sigmoid_derivative(z_values)

plt.figure(figsize=(8, 5))
plt.plot(z_values, sigma_values, label="σ(z)", color="blue")
plt.plot(z_values, deriv_values, label="σ'(z)", color="red", linestyle='dashed')
plt.title("Fonction Sigmoïde et sa Dérivée")
plt.xlabel("z")
plt.ylabel("Valeur")
plt.legend()
plt.grid()
# plt.show()
plt.savefig("graphique.png")

# b : verification
np.allclose(deriv_values, sigma_values * (1 - sigma_values))