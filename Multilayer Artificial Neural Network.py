from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# MNIST dataset: training data consist of handwritten digits from 250 different people, 50% school students,
# 50% employees from Census Bureau. Test dataset contains handwritten digits from different people same split.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

print(X.shape)
print(y.shape)

X = ((X / 225.) - .5) * 2

fig, ax = plt.subplots(nrows=2, ncols=5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10): # show each unique digit representation
    img = X[y == i][0].reshape(28, 28)
    #print(X[y == i][0].reshape(28, 28))
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

"""
fig, ax = plt.subplots(nrows=5, ncols=5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25): # see first 25 variants of digit 7
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
"""

# Divide dataset into training, validation and test subsets
# Following will split dataset such that 55000 images are used for training, 5000 for images for validation and
# 10000 images for testing
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

##########################
### MODEL
##########################

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

class NeuralNetMLP:

    # constructor instantiates the weight matrices and bias vector for the hidden and output layer.
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        # hidden
        rng =  np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        # output
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    # forward method takes in one or more training examples and returns the predictions.
    # meaning, both activation values from the hidden layer and the output layer, a_h and a_out.
    # while a_out represents the class-membership probabilities that we can convert to class labels,
    # we also need activation values from the hidden layer a_h to optimize the model parameters.
    # that is; the weight and bias units of the hidden and output layers.
    def forward(self, x):
        # Hidden Layer

        # input dim: [n_hidden, n_features]
        #       dot [n_features, n_examples].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output Layer

        # input dim: [n_classes, n_hidden]
        #       dot [n_hidden, n_examples].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        #########################
        ### Output layer weights
        #########################

        # one-hot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use

        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # gradient for output weights

        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h

        # input dim: [n_classes, n_examples]
        #       dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        #################################
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet
        # * dHiddenNet/dWeight

        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out

        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative

        # [n_examples, n_features]
        d_z_h__d_w_h = x

        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
                               d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)

# instantiate new NNMLP
# Model accepts MNIST images reshaped into 784-dimensional vectors (in the X_train, X_valid, X_test format)
# for 10 integer classes (digits 0-9), and consisting of 50 hidden layer nodes.
model = NeuralNetMLP(num_features=28*28,
                     num_hidden=50,
                     num_classes=10)

# Coding the neural network training loop

# Defining data loaders:

num_epochs = 50
minibatch_size = 100

# Function taking our dataset and divides it into mini-batches of desired size for stochastic gradient descent training
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    #print(indices)
    np.random.shuffle(indices)
    #print("SHOOOOOO")
    #print(indices.shape[0])
    for start_idx in range(0, indices.shape[0] - minibatch_size
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# iterate over training epochs
for i in range(num_epochs):
    # iterate over minibatches
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break
print(X_train_mini.shape)
print(y_train_mini.shape)


# Defining a function to compute the loss and accuracy
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)
def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
print(f'Initial validation MSE: {mse:.1f}')

predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)
print(f'Initial validation accuracy: {acc*100:.1f}%')


# Compute MSE and accuracy incrementally by iterating over the dataset one mini-batch at a time
# to be more memory-efficient, due to large matrix multiplication using forward pass.
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)

            #### Comptue gradients ####
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini,
                                                                                            a_h, a_out,
                                                                                            y_train_mini)
            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        #### Epoch Logging ####
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc

# train our model for 50 epochs
np.random.seed(123) # for the training set shuffling
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.1)

# In general training Deep Neural Networks are relatively expensive compared to other models.
# So we stop it early in certain circumstances and start over with different hyperparameter settings.
# If we find that it increasingly tends to overfit the training data (noticeable by an increasing gap between
# training and validation dataset performance), we may want to stop the training early, as well.


# ## Evaluating the neural network performance
# in train() we collected training loss and training validation accuracy for each epoch
# so that we can visualize the results using matplotlib.

# Plot of training MSE loss:
plt.plot(range(len(epoch)), epoch_loss)
plt.ylabel('Mean squared error')
plt.xlabel('Epoch')
plt.show()

# loss is decreased substantially during first 10 epochs, and seems to slowly converge in the last 10 epochs
# However the small slope between epoch 40 and 50 indicates  that the loss would further decrease with training over
# additional epochs.

# Training and validation accuracy:
plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(loc='lower right')
plt.show()

# preceding code examples plot those accuracy values over the 50 training epochs
# plot reveals that the gap between training and validation accuracy increases as we train for more epochs
# At approx. the 25th epoch, the training and validation accuracy values are almost equal and then the
# network starts to slightly overfit the training data.

# One way to decrease the effect of overfitting is to increase the regularization strength via L2 regularization
# Another useful technique for tackling overfitting in NNs is dropout

# Finally evaluate the generalization performance of the model by calculating the prediction accuracy on the test dataset:
test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

# We can see the test accuracy is very close to the validation set accuracy corresponding to the last epoch (94.74%).
# Moreover the respective training accuracy is only minimally higher at 95.59%, reaffirming that our model only
# slightly overfits the training data.

# To further fine-tune the model, we could change the number of hidden units, the learning rate or use various other
# tricks that have been developer over the years but are beyond the scope of this book.

# Additional performance-enhancing tricks such as adaptive learning rates, more sophisticated SGD-based optimization
# algorithms, batch normalization and dropout.
# Other common tricks beyond the scope of the following chapters include:
# - Adding skip-connection which are the main contribution of residual NNs
# - Using learning rate schedulers that change the learning rate during training
# - Attaching loss functions to earlier layers in the networks


# Lastly we take a look at images that our MLP struggles with by extracting and plotting
# the first 25 misclassified samples from the test set:
X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]
_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)
misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()
for i in range(25):
    img = misclassified_images[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {correct_labels[i]}\n'
                    f' Predicted: {misclassified_labels[i]}')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# We should now see a 5x5 subplot matrix where the first number in the subtitles indicates the plot index
# the second number represents the true class label (True), the third number stand for predicted class label (Predicted):
# As seen the network finds 7s challenging, when they include a horizontal line as in  examples 19 and 20.
# Looking back at an earlier figure in this chapter where we plotted different training examples of the number 7,
# we can hypothesize that the handwritten digit 7 with a horizontal line is underrepresented in our dataset and often misclassified

