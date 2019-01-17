import numpy as np
import sampler as sp
import matplotlib.pyplot as plt


# ap = [0.1,0.5,0.2,0.2]
# categorical = sp.Categorical(ap)
# samples = np.zeros((len(ap),))
# for i in range(10000):
#     p = categorical.sample()
#     samples[p] += 1

# plt.bar([0,1,2,3],samples)
# plt.xlabel("x")
# plt.ylabel("numbers")
# plt.title("Categorical Distribution")
# plt.savefig("fig1.png")
#plt.show()

# mu = 5
# sigma = 2
# size = 100000
# uni_normal = sp.UnivariateNormal(mu, sigma)
# samples = uni_normal.pmf(size)
# plt.hist(samples, bins=300)
# plt.title("Univariate Normal Distribution")
# plt.xlabel("x")
# plt.ylabel("numbers")
# plt.savefig("fig2.png")
# plt.show()



# Mu = np.array([2,3])
# Sigma = np.array([[1, 0.5], [0.2, 1]])
# multi_normal = sp.MultiVariateNormal(Mu, Sigma)
# samples = multi_normal.pmf(1000)
# x = samples[:,0]
# y = samples[:,1]
# plt.scatter(x, y)
# plt.title("2D Multivariate Normal Distribution")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("fig3.png")
# #plt.show()


ap = np.array([0.25,0.25,0.25,0.25])
Mu = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
Sigma = np.array([[1, 0], [0, 1]])
pm = [
    sp.MultiVariateNormal(Mu[0], Sigma),
    sp.MultiVariateNormal(Mu[1], Sigma),
    sp.MultiVariateNormal(Mu[2], Sigma),
    sp.MultiVariateNormal(Mu[3], Sigma)
]
mixture = sp.MixtureModel(ap, pm)
size = 10000
x = np.zeros((size,))
y = np.zeros((size,))
cnt = 0
for i in range(size):
    t = mixture.sample()
    x[i] = t[0]
    y[i] = t[1]
    if ((t[0]-0.1)**2 + (t[1]-0.2)**2 <= 1):
        cnt += 1

print("The probability is", 1.0 * cnt / size)
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("mixture model")
#plt.show()
plt.savefig("fig4.png")


