import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):
	def __init__(self):
		super().__init__()
		# flatten images into usable data
		# basically an input with [a, b, c, d] dimensions turns into output with [a, bcd] dimensions
		# [[1, 1, 1]]
		self.flatten = nn.Flatten() 

		# define neural net layers
		self.dense_layers = nn.Sequential(
			nn.Linear(28*28, 256),
			nn.ReLU(),
			nn.Linear(256, 10)
		)

		# standardize output
		self.softmax = nn.Softmax(dim=1)

	def forward(self, input_data):
		flattened_data = self.flatten(input_data)
		logits = self.dense_layers(flattened_data)
		predictions = self.softmax(logits)
		return predictions


def download_mnist_datasets():
	train_data = datasets.MNIST(
		root="data",
		download=True,
		train=True,
		transform=ToTensor()
	)
	validation_data = datasets.MNIST(
		root="data",
		download=True,
		train=False,
		transform=ToTensor()
	)
	return train_data, validation_data

def train_one_epoch(model, data_loader, loss_function, optimiser, device):
	for inputs, targets in data_loader:
		inputs, targets = inputs.to(device), targets.to(device)
		
		# calculate loss
		predictions = model(inputs)
		loss = loss_function(predictions, targets)

		# back propogate loss and update weights
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()

	print(f"loss = {loss.item()}")

def train(model, data_loader, loss_function, optimiser, device, epochs):
	for i in range(epochs):
		print(f"Epoch = {i+1}")
		train_one_epoch(model, data_loader, loss_function, optimiser, device)
		print("---------------")
	print("Training finished!")

if __name__ == "__main__":
	train_data, validation_data = download_mnist_datasets()
	print("MNIST dataset downloaded")

	train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

	if torch.cuda.is_available():
		device = "cuda"
	else:
		device = "cpu"
	print(f"Using {device} device")
	feed_forward_net = FeedForwardNet().to(device)

	loss_fn = nn.CrossEntropyLoss()
	optimiser = torch.optim.Adam(feed_forward_net.parameters(), 
							  lr=LEARNING_RATE)

	train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

	torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
	print("Model trained and stored at feedforwardnet.pth")

