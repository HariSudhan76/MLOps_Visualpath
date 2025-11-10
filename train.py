import argparse

def train_model(data_path,epochs, lr):
	print(f"Training model on {data_path} for {epochs} epochs with lr={lr}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="ML Model Training via CLI")
	parser.add_argument("--data_path",type=str, required=True)
	parser.add_argument("--epochs",type=int,  default=10)
	parser.add_argument("--lr",type=float, default=0.001)
	
	args = parser.parse_args()
	
	train_model(args.data_path, args.epochs, args.lr)